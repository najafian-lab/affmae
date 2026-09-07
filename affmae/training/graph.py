"""CUDA graph replay for fixed-shape training microbatches."""

import logging

import torch


class TrainingGraph:
    """Capture forward/loss/backward; keep optimizer and scaler updates eager."""

    def __init__(self, model, enabled=True):
        self.model = model
        device = next(model.parameters()).device
        self.enabled = bool(
            enabled and device.type == "cuda" and torch.version.hip is None
            and not isinstance(model, torch.nn.parallel.DistributedDataParallel)
        )
        self.graph = None
        self.shape_warning = False
        self.one = torch.ones((), device=device)

    @staticmethod
    def _signature(inputs):
        return tuple((t.shape, t.dtype, t.device) for t in inputs)

    def _capture(self, step, inputs):
        model = self.model
        device = inputs[0].device
        self.inputs = tuple(t.detach().clone() for t in inputs)
        self.signature = self._signature(inputs)
        # Warmup must not consume training RNG or update BatchNorm statistics.
        buffers = [(b, b.detach().clone()) for b in model.buffers()]
        side = torch.cuda.Stream(device=device)
        current = torch.cuda.current_stream(device)
        side.wait_stream(current)
        try:
            with torch.random.fork_rng(devices=[device]):
                with torch.cuda.stream(side):
                    for _ in range(3):
                        model.zero_grad(set_to_none=True)
                        step(*self.inputs)
                current.wait_stream(side)
                self.parameters = [p for p in model.parameters() if p.grad is not None]
                model.zero_grad(set_to_none=True)
                self.gradients = [torch.zeros_like(p) for p in self.parameters]
                for p, grad in zip(self.parameters, self.gradients):
                    p.grad = grad
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=side):
                    self.output = step(*self.inputs)
                self.zero_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.zero_graph, stream=side):
                    torch._foreach_zero_(self.gradients)
                current.wait_stream(side)
                self.graph = graph
        finally:
            current.wait_stream(side)
            with torch.no_grad():
                for buffer, saved in buffers:
                    buffer.copy_(saved)
            model.zero_grad(set_to_none=True)

    def run(self, step, inputs, scaler):
        """Run one microbatch; ``step`` accepts inputs followed by a GPU loss scale.

        ``step`` performs backward and returns tensors used for logging. Callers
        clear gradients only at accumulation boundaries. Shape changes run eagerly.
        """
        # Public GradScaler API initializes its state and supplies the current scale.
        inputs = (*inputs, self.one if scaler is None else scaler.scale(self.one))
        if not self.enabled:
            return step(*inputs)
        if self.graph is None:
            # Capture starts only at a clean accumulation boundary.
            if any(p.grad is not None for p in self.model.parameters()):
                return step(*inputs)
            try:
                self._capture(step, inputs)
            except torch.OutOfMemoryError:
                self.enabled = False
                self.graph = None
                for name in ("inputs", "output", "gradients", "parameters", "zero_graph"):
                    self.__dict__.pop(name, None)
                logging.warning("CUDA graph capture ran out of memory; using eager training.")
                torch.cuda.empty_cache()
                return step(*inputs)
        if self._signature(inputs) != self.signature:
            if not self.shape_warning:
                logging.info("Changed microbatch shape: using eager execution for this shape.")
                self.shape_warning = True
            return step(*inputs)
        for dst, src in zip(self.inputs, inputs):
            dst.copy_(src)
        # Stable gradient storage accumulates until the optimizer clears it.
        if all(p.grad is None for p in self.parameters):
            self.zero_graph.replay()
        else:
            for p, grad in zip(self.parameters, self.gradients):
                if p.grad is None:
                    grad.zero_()
                elif p.grad is not grad:
                    grad.copy_(p.grad)
        for p, grad in zip(self.parameters, self.gradients):
            p.grad = grad
        self.graph.replay()
        return self.output
