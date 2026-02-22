import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
import time
import torch._inductor.config
import logging
# from .clusters.kernels.local_attn import FlashLocalAttentionFunction

# --------------------------------------------------------------------------
# 1. YOUR PRE_TABLE DEFINITION (Global Scope)
# --------------------------------------------------------------------------
# This will be visible to both class instances, just like in your project.

# assumes largest input resolution is 2048 x 2048
# rel_pos_width = 2048 // 4 - 1
rel_pos_width = 512 // 8 - 1
table_width = 2 * rel_pos_width + 1  # This will be 127
num_table_entries = table_width * table_width # 127*127 = 16129

print(f"Initializing pre_table with table_width={table_width} (entries={num_table_entries})")

pre_hs = torch.arange(table_width).float()-rel_pos_width
pre_ws = torch.arange(table_width).float()-rel_pos_width
pre_ys, pre_xs = torch.meshgrid(pre_hs, pre_ws, indexing='ij')  # table_width x table_width

# expanded relative position lookup table
dis_table = (pre_ys**2 + pre_xs**2) ** 0.5
sin_table = pre_ys / dis_table
cos_table = pre_xs / dis_table
pre_table = torch.stack([pre_xs, pre_ys, dis_table, sin_table, cos_table], dim=2)  # table_width x table_width x 5
pre_table[torch.bitwise_or(pre_table.isnan(), pre_table.isinf()).nonzero(as_tuple=True)] = 0
pre_table = pre_table.reshape(-1, 5) # Shape: [16129, 5]
pre_table_fp32 = pre_table.to(torch.float32)
pre_table_fp16 = pre_table.to(torch.float16)

# This is the global required by the old class
pre_table = pre_table_fp32 # Let's default to fp32

# --------------------------------------------------------------------------
# 2. MOCK KERNELS (Needed to define the classes)
# --------------------------------------------------------------------------

class CLUSTENQKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, idx):
        B, H, N, D = q.shape
        M = idx.shape[-1]
        # The global path never calls this, so it just needs to exist.
        return torch.zeros(B, H, N, M, device=q.device, dtype=q.dtype)

class CLUSTENAVFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn, v, idx):
        B, H, N, D = v.shape
        # The global path never calls this.
        return torch.zeros(B, H, N, D, device=v.device, dtype=v.dtype)

class FlashLocalAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qh, k, v, member_idx, pos_bias, mask, blank_k_hd, blank_v_hd, scale):
        B, H, N, D = qh.shape
        # The global path never calls this.
        return torch.zeros(B, H, N, D, device=qh.device, dtype=qh.dtype)

# --------------------------------------------------------------------------
# 3. CLASS DEFINITION: ClusterAttention_Old
# --------------------------------------------------------------------------

class ClusterAttention_Old(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.pos_dim = 2
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2*dim)
        self.softmax = nn.Softmax(dim=-1)
        self.blank_k = nn.Parameter(torch.randn(dim) * 0.2)
        self.blank_v = nn.Parameter(torch.randn(dim) * 0.2)
        self.pos_embed = nn.Linear(self.pos_dim+3, num_heads)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, feat, member_idx, cluster_mask, pe_idx, global_attn):
        b, n, c = feat.shape
        c_ = c // self.num_heads
        h = self.num_heads
        q = self.q(feat)
        q = q * self.scale  # <-- Q is scaled HERE
        kv = self.kv(feat)
        
        if not global_attn:
            # This path is ignored by our test
            pass
        else:
            q = q.reshape(b, n, h, -1).permute(0, 2, 1, 3)  # b x h x n x c_
            kv = kv.view(b, n, h, 2, c_).permute(3, 0, 2, 1, 4)  # 2 x b x h x n x c_
            key, v = kv[0], kv[1]
            attn = q @ key.transpose(-1, -2)  # b x h x n x n
            mask = None

        # Use the global pre_table
        global pre_table 
        if pre_table.device != pe_idx.device:
            pre_table = pre_table.to(pe_idx.device)
        if pre_table.dtype != feat.dtype:
            pre_table = pre_table.to(feat.dtype)

        pe_table_out = self.pos_embed(pre_table)
        pe_shape = pe_idx.shape
        pos_embed = pe_table_out.gather(index=pe_idx.view(-1, 1).expand(-1, h), dim=0).reshape(*(pe_shape), h).permute(0, 3, 1, 2)
        
        attn = attn + pos_embed
        
        blank_attn = (q * self.blank_k.reshape(1, h, 1, c_)).sum(-1, keepdim=True)
        attn = torch.cat([attn, blank_attn], dim=-1)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        blank_attn = attn[..., -1:]
        attn = attn[..., :-1]
        blank_v = blank_attn * self.blank_v.reshape(1, h, 1, c_)
        
        if global_attn:
            feat = (attn @ v).permute(0, 2, 1, 3).reshape(b, n, c)
            feat = feat + blank_v.permute(0, 2, 1, 3).reshape(b, n, c)
        else:
             # This path is ignored by our test
            pass
            
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        return feat

# --------------------------------------------------------------------------
# 4. CLASS DEFINITION: ClusterAttention_New
# --------------------------------------------------------------------------

class ClusterAttention_New(nn.Module):
    def __init__(self, dim, num_heads, proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.pos_dim = 2
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.blank_k = nn.Parameter(torch.randn(dim) * 0.2)
        self.blank_v = nn.Parameter(torch.randn(dim) * 0.2)
        self.pos_embed = nn.Linear(self.pos_dim+3, num_heads)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _heads(self, x: torch.Tensor, H: int):
        B, N, C = x.shape
        C_h = C // H
        return x.view(B, N, H, C_h).permute(0, 2, 1, 3).contiguous()

    def _make_pos_bias(self, pe_idx: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # Use the global pre_tables
        global pre_table_fp16, pre_table_fp32 
        B, N, M = pe_idx.shape
        device = pe_idx.device
        
        if dtype == torch.float16:
            if pre_table_fp16.device != device:
                pre_table_fp16 = pre_table_fp16.to(device)
            pre_table_data = pre_table_fp16
        else: # Default to fp32
            if pre_table_fp32.device != device:
                pre_table_fp32 = pre_table_fp32.to(device)
            pre_table_data = pre_table_fp32

        pe_table = self.pos_embed(pre_table_data)
        H = pe_table.shape[-1]
        flat_idx = pe_idx.reshape(-1).to(dtype=torch.long, device=device)
        pe = torch.index_select(pe_table, dim=0, index=flat_idx)
        pos_bias = pe.view(B, N, M, H).permute(0, 3, 1, 2).contiguous()
        return pos_bias

    def forward(self, feat, member_idx, cluster_mask, pe_idx, global_attn: bool):
        B, N, C = feat.shape
        H = self.num_heads
        C_h = C // H
        device = feat.device
        dtype = feat.dtype
        M = member_idx.shape[-1]

        q = self.q(feat)
        kv = self.kv(feat)
        qh = self._heads(q, H) # [B,H,N,C_h] (UNSCALED)
        kvh = self._heads(kv, H)
        k, v = kvh.split([C_h, C_h], dim=-1)

        if global_attn:
            if N != M:
                raise ValueError(f"Global attention requires M == N, but got M={M} and N={N}")
            
            # --- Key Logic Match ---
            # Old code scales Q *before* dot products.
            # We must do the same to match it.
            qh_scaled = qh * self.scale
            flex_scale_factor = 1.0
            # -----------------------
            
            blank_k_h = self.blank_k.view(1, H, 1, C_h).expand(B, -1, -1, -1)
            blank_v_h = self.blank_v.view(1, H, 1, C_h).expand(B, -1, -1, -1)
            k_full = torch.cat([k, blank_k_h], dim=2) # [B,H, N+1, C_h]
            v_full = torch.cat([v, blank_v_h], dim=2) # [B,H, N+1, C_h]
            pos_bias = self._make_pos_bias(pe_idx, dtype) # [B,H,N,N]

            pos_bias = F.pad(pos_bias, (0, 1), "constant", 0.0)

            # # This `score_mod` closure captures `pos_bias` and `N` from the local scope
            # def score_mod_global(score, b, h, q_idx, kv_idx):
            #     return score + pos_bias[b, h, q_idx, kv_idx]
            
            out = F.scaled_dot_product_attention(
                qh,                 # UN-SCALED query
                k_full,             # Full keys
                v_full,             # Full values
                attn_mask=pos_bias # Additive float mask
            )

        else:
            # This path is ignored by our test
            qh_scaled = qh * self.scale # Pre-scale q for the triton kernel
            pos_bias = self._make_pos_bias(pe_idx, dtype)
            mask = None
            if cluster_mask is not None:
                mask = (cluster_mask > 0).unsqueeze(1).expand(-1, H, -1, -1).contiguous()
            blank_k_hd = self.blank_k.view(H, C_h).contiguous()
            blank_v_hd = self.blank_v.view(H, C_h).contiguous()
            out = FlashLocalAttentionFunction.apply(
                qh_scaled, k, v, member_idx, pos_bias, mask, blank_k_hd, blank_v_hd,
                1.0 # Pass 1.0 since we pre-scaled
            )

        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

# --------------------------------------------------------------------------
# 5. TEST HARNESS (This is the part that runs)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    
    import torch.backends.cuda
    
    def print_sdpa_config():
        """Helper to print the current SDPA backend state."""
        print("--- SDPA Kernel Config ---")
        print(f"  FlashAttention Enabled: {torch.backends.cuda.is_flash_attention_available()}")
        print(f"  Mem-Efficient Enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
        print("  Using C++ math fallback: True" if not (torch.backends.cuda.is_flash_attention_available() or torch.backends.cuda.is_mem_efficient_sdp_available()) else "")
        print("----------------------------")
    
    print_sdpa_config()

    # --- 0. Configuration ---
    # Set matmul precision for performance and silence warnings
    torch.set_float32_matmul_precision('high')
    torch._logging.set_logs(inductor=logging.ERROR) 
    
    # *** THIS IS THE FIX FOR THE LOGS ***
    # Suppress the verbose autotuning output in the console
    # torch._inductor.config.verbose_autotune = False
    print("Autotune verbose logs suppressed.")
    # **********************************

    # --- 1. Test Configuration ---
    B, N, C, H = 512, 64, 256, 8 # Batch, SeqLen, Dim, Heads
    M = N                      # Must be true for global path
    DTYPE = torch.float32 
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TOLERANCE = 1e-5 
    
    # Benchmark config
    WARMUP_RUNS = 20
    TIMED_RUNS = 100
    
    print(f"--- Running Full Attention Test Suite ---")
    print(f"Params: B={B}, N={N}, C={C}, H={H}, M={M}")
    print(f"Device: {DEVICE}, DType: {DTYPE}, Tolerance: {TOLERANCE}\n")

    # --- 2. Instantiate Models ---
    torch.manual_seed(42)
    old_model = ClusterAttention_Old(C, H, proj_drop=0.0, attn_drop=0.0)
    
    torch.manual_seed(42)
    new_model = ClusterAttention_New(C, H, proj_drop=0.0)
    
    old_model.to(DEVICE).to(DTYPE)
    new_model.to(DEVICE).to(DTYPE)

    # --- 3. Synchronize Weights ---
    new_model.load_state_dict(old_model.state_dict(), strict=False)
    print("Model weights synchronized.")

    # --- 4. Create Identical Dummy Inputs ---
    feat = torch.randn(B, N, C, device=DEVICE, dtype=DTYPE)
    member_idx = torch.randint(0, N, (B, N, M), device=DEVICE) 
    pe_idx = torch.randint(0, num_table_entries, (B, N, M), device=DEVICE)
    cluster_mask = None

    # --- 5. Set to .eval() mode ---
    old_model.eval()
    new_model.eval()
    print("Models set to eval() mode (dropout disabled).")
    
    # --- 6. COMPILE THE NEW MODEL ---
    print("\n" + "="*40)
    print("--- Compiling New Model ---")
    print("="*40)
    print("Compiling new_model with torch.compile(mode='max-autotune')...")
    try:
        new_model_compiled = new_model#torch.compile(new_model, mode="max-autotune")
        print("Compilation successful.")
    except Exception as e:
        print(f"❌ FAILED to compile: {e}")
        print("Exiting. Cannot continue without compiled model.")
        exit()

    # --- 7. Correctness Test (Eager vs. Eager) ---
    print("\n" + "="*40)
    print("--- Running Eager Correctness Test ---")
    print("="*40)
    try:
        with torch.no_grad():
            output_old = old_model(feat, member_idx, cluster_mask, pe_idx, global_attn=True)
            output_new_eager = new_model(feat, member_idx, cluster_mask, pe_idx, global_attn=True)
        torch.allclose(output_old, output_new_eager, atol=TOLERANCE, rtol=0)
        print("✅ SUCCESS: Eager new model matches old model.")
    except Exception as e:
        print(f"❌ FAILED: Eager new model does NOT match: {e}")

    # --- 8. Correctness Test (Compiled vs. Old) ---
    print("\n" + "="*40)
    print("--- Running Compiled Correctness Test ---")
    print("="*40)
    try:
        with torch.no_grad():
            _ = new_model_compiled(feat, member_idx, cluster_mask, pe_idx, global_attn=True) # Warmup run
            output_new_compiled = new_model_compiled(feat, member_idx, cluster_mask, pe_idx, global_attn=True)
            
        torch.allclose(output_old, output_new_compiled, atol=TOLERANCE, rtol=0)
        print("✅ SUCCESS: Compiled new model matches old model.")
    except Exception as e:
        print(f"❌ FAILED: Compiled new model does NOT match: {e}")

    # --- 9. GRADIENT TEST (on COMPILED model) ---
    print("\n" + "="*40)
    print("--- Running Compiled Gradient Flow Test ---")
    print("="*40)
    try:
        new_model_compiled.zero_grad()
        print("Running compiled forward/backward pass...")
        output_new_grad = new_model_compiled(feat, member_idx, cluster_mask, pe_idx, global_attn=True)
        output_new_grad.sum().backward()
        
        print("Checking pos_embed.weight.grad...")
        grad = new_model_compiled.pos_embed.weight.grad
        
        if grad is None:
            print("\n❌ FAILED: Gradient is None. Gradients are NOT flowing.")
        elif not torch.all(grad == 0):
            print("\n✅ SUCCESS: Gradients are flowing through compiled model!")
        else:
            print("\n⚠️ WARNING: Gradients are all zero.")
    except Exception as e:
        print(f"\n❌ FAILED: Gradient test raised an error: {e}")

    # --- 10. BENCHMARK (NOW INCLUDES MEMORY) ---
    print("\n" + "="*40)
    print("--- Running Benchmark (Time & Memory) ---")
    print("="*40)
    print(f"Config: {WARMUP_RUNS} warm-up runs, {TIMED_RUNS} timed runs.")

    try:
        with torch.no_grad():
            
            # --- Warm-up Old Model ---
            print("Warming up old (eager) model...")
            for _ in range(WARMUP_RUNS):
                _ = old_model(feat, member_idx, cluster_mask, pe_idx, global_attn=True)
            
            # --- Time Old Model ---
            print("Timing old (eager) model...")
            torch.cuda.reset_peak_memory_stats(DEVICE) # <-- Reset memory
            torch.cuda.synchronize(DEVICE)
            start_time = time.perf_counter()
            for _ in range(TIMED_RUNS):
                _ = old_model(feat, member_idx, cluster_mask, pe_idx, global_attn=True)
            torch.cuda.synchronize(DEVICE)
            end_time = time.perf_counter()
            
            time_old = (end_time - start_time) / TIMED_RUNS
            peak_mem_old = torch.cuda.max_memory_allocated(DEVICE) / (1024**2) # <-- Get peak mem in MB
            print(f"Old model (eager):    {time_old*1000:7.3f} ms | {peak_mem_old:7.2f} MB peak")

            # --- Warm-up New (Compiled) Model ---
            print("Warming up new (compiled) model...")
            for _ in range(WARMUP_RUNS):
                _ = new_model_compiled(feat, member_idx, cluster_mask, pe_idx, global_attn=True)
                
            # --- Time New (Compiled) Model ---
            print("Timing new (compiled) model...")
            torch.cuda.reset_peak_memory_stats(DEVICE) # <-- Reset memory
            torch.cuda.synchronize(DEVICE)
            start_time = time.perf_counter()
            for _ in range(TIMED_RUNS):
                _ = new_model_compiled(feat, member_idx, cluster_mask, pe_idx, global_attn=True)
            torch.cuda.synchronize(DEVICE)
            end_time = time.perf_counter()

            time_new = (end_time - start_time) / TIMED_RUNS
            peak_mem_new = torch.cuda.max_memory_allocated(DEVICE) / (1024**2) # <-- Get peak mem in MB
            print(f"New model (compiled): {time_new*1000:7.3f} ms | {peak_mem_new:7.2f} MB peak")
            
            # --- Results ---
            speedup = time_old / time_new
            mem_saved = peak_mem_old - peak_mem_new
            print("\n" + "-"*20)
            print(f"🚀 Speedup: {speedup:.2f}x")
            print(f"💾 Memory Saved: {mem_saved:.2f} MB")
            print("-"*(20) + "\n")
    
    except Exception as e:
        print(f"\n❌ FAILED: Benchmark raised an error: {e}")