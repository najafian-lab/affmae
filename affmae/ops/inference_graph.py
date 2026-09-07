"""Bounded CUDA graph replay for public segmentation inference."""

import logging
from threading import RLock

import torch
from torch.utils._pytree import tree_map

logger = logging.getLogger(__name__)


class InferenceGraph:
    def __init__(self, model, enabled=True):
        self.model = model
        self.enabled = enabled
        self.entries = {}
        self.failed = set()
        self.stream = None
        self.lock = RLock()

    def clear(self):
        """Release captures before replacing model parameters or configuration."""
        with self.lock:
            if self.stream is not None:
                self.stream.synchronize()
            self.stream = None
            self.entries.clear()
            self.failed.clear()

    def _capture(self, step, images):
        static = torch.empty_like(images)
        static.copy_(images)
        buffers = [(b, b.clone()) for b in self.model.buffers()]
        try:
            with torch.random.fork_rng(devices=[images.device]):
                for _ in range(3):
                    step(static)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self.stream):
                    output = step(static)
        finally:
            for buffer, saved in buffers:
                buffer.copy_(saved)
        return graph, static, output

    def run(self, step, images, stages=False):
        """Return owned tensors; later replays cannot overwrite earlier results."""
        if (not self.enabled or images.device.type != "cuda" or torch.version.hip
                or self.model.training
                or isinstance(self.model, (torch.nn.DataParallel,
                                          torch.nn.parallel.DistributedDataParallel))
                or torch.cuda.is_current_stream_capturing()):
            output = step(images)
            return output if stages else output[0]
        key = (tuple(images.shape), tuple(images.stride()), images.dtype, images.device,
               torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"),
               torch.get_float32_matmul_precision(),
               torch.backends.cudnn.allow_tf32, torch.backends.cuda.flash_sdp_enabled(),
               torch.backends.cuda.mem_efficient_sdp_enabled(),
               torch.backends.cuda.math_sdp_enabled())
        with self.lock, torch.cuda.device(images.device):
            if key in self.failed or (key not in self.entries and len(self.entries) >= 2):
                output = step(images)
                return output if stages else output[0]
            if self.stream is None:
                self.stream = torch.cuda.Stream(device=images.device)
            current = torch.cuda.current_stream(images.device)
            self.stream.wait_stream(current)
            try:
                with torch.cuda.stream(self.stream):
                    images.record_stream(self.stream)
                    if key not in self.entries:
                        try:
                            self.entries[key] = self._capture(step, images)
                        except RuntimeError as error:
                            if any(text in str(error).lower() for text in
                                   ("illegal memory access", "device-side assert")):
                                raise
                            self.failed.add(key)
                            logger.warning("Inference graph capture failed; using eager: %s", error)
                    if key in self.entries:
                        graph, static, output = self.entries[key]
                        static.copy_(images)
                        # CUDA graph replay reduces launch overhead.
                        graph.replay()
                        selected = output if stages else output[0]
                        result = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x,
                                          selected)
                    else:
                        output = step(images)
                        result = output if stages else output[0]
            finally:
                current.wait_stream(self.stream)
            return result
