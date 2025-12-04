---
title: Hybrid Attention Mechanism Options
draft: false
tags:
  - NLP
  - Attention
  - Hybrid
  - Study-Notes
  - KDA
  - DSA
date: 2025-12-05
---

## 🔬 Kimi Delta Attention (KDA) + DeepSeek Sparse Attention (DSA)

Hybrid Transformer Attention Layers - Advanced Implementation & Comparison

Based on arXiv:2510.26692 (Kimi Linear) and DeepSeek-V3.2-Exp Architecture

### Kimi Delta Attention (KDA)

> [!info]
> **Linear Complexity:** $O(N)$ time & memory
> **Core:** Channel-wise gated DeltaNet with DPLR-style decay

#### KDA-Mechanism

1. **Feature Projection:** Input $X$ is projected to $Q, K, V$ and a data-dependent gate $\gamma$.
2. **Delta Rule Update:** Maintain a recurrent state $S_t \in \mathbb{R}^{d \times d}$.
   Unlike standard Linear Attention, KDA uses a **channel-wise forget gate** $\gamma_t \in \mathbb{R}^d$ to control memory retention per feature.
3. **Chunkwise Computation:** Processes tokens in chunks (e.g., 64) to leverage Tensor Cores, balancing recurrence and parallel matrix multiplication.

##### Key Improvement

* ✓ **Finer-grained Gating:** Element-wise control over state decay allows "forgetting" irrelevant context selectively.
* ✓ **Hardware Efficiency:** Specialized DPLR (Diagonal Plus Low Rank) kernel formulation.

#### KDA-Math

$$ \mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t, \mathbf{\beta}_t = \text{Proj}(\mathbf{x}_t) $$
$$ \mathbf{g}_t = \sigma(\mathbf{\beta}_t) \quad \text{(Data-Dependent Decay)} $$
$$ \mathbf{S}_t = \mathbf{S}_{t-1} \odot \mathbf{g}_t + \mathbf{k}_t^\top \mathbf{v}_t $$
$$ \mathbf{o}_t = \text{LayerNorm}(\mathbf{q}_t \mathbf{S}_t) $$

*Note: The decay rate $\mathbf{g}_t$ is **data-dependent** (computed from input $x_t$) and applied **channel-wise** (broadcasting across the $D \times D$ state). This differentiates KDA from standard Linear Attention or simple DeltaNet.*

#### KDA-Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.jit.script
class KimiDeltaAttention(nn.Module):
    """
    Corrected KDA: Gated DeltaNet with Data-Dependent Decay
    Ref: arXiv:2510.26692 (Kimi Linear)
    """
    def __init__(self, dim: int, num_heads: int = 8, chunk_size: int = 64):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.chunk_size = chunk_size
        
        # Projections: Q, K, V, and Beta (Decay/Gate)
        # Beta is derived from input x, making it data-dependent
        self.qkv_gate = nn.Linear(dim, 3 * dim + dim, bias=True)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor, state: torch.Tensor = None):
        B, N, D = x.shape
        H, HD = self.num_heads, self.head_dim
        
        # 1. Project inputs
        proj = self.qkv_gate(x)
        q, k, v, beta = torch.split(proj, [D, D, D, D], dim=-1)
        
        # 2. Reshape & Activation
        q = q.view(B, N, H, HD)
        k = k.view(B, N, H, HD)
        v = v.view(B, N, H, HD)
        # Decay rate g in (0, 1)
        g = torch.sigmoid(beta.view(B, N, H, HD)) 
        
        # 3. Initialize State (B, H, D, D)
        if state is None:
            state = torch.zeros(B, H, HD, HD, device=x.device)
            
        outputs = []
        
        # 4. Chunkwise Recurrence
        # (Simplified loop; real impl uses DPLR kernels)
        for t in range(N):
            q_t = q[:, t]
            k_t = k[:, t]
            v_t = v[:, t]
            g_t = g[:, t].unsqueeze(-1) # Broadcast over last dim
            
            # State Update: S_t = S_{t-1} * g_t + K^T * V
            state = state * g_t + torch.einsum('bhd,bhm->bhdm', k_t, v_t)
            
            # Output: O_t = Q_t * S_t
            out_t = torch.einsum('bhd,bhdm->bhm', q_t, state)
            outputs.append(out_t)
            
        output = torch.stack(outputs, dim=1).reshape(B, N, D)
        return self.out_proj(output), state
```

### DeepSeek Sparse Attention (DSA)

> [!info]
> **Sparsity:** Top-K Selection (e.g., $k=64$)
> **Core:** Lightning Indexer (Query-Dependent) + Sparse Gather

#### DSA-Mechanism

1. **Lightning Indexer:** A lightweight attention branch. It projects inputs to compressed/quantized keys (often FP8) to quickly score relevance.
2. **Query-Dependent Scoring:** Unlike static scoring, the indexer computes scores between the *current query* and *all past compressed keys*.
3. **Top-K Selection:** Selects the top-k indices with highest scores.
4. **Sparse Attention:** Fetches full-precision KV pairs for these indices and performs standard attention.

> [!warning]
> **Note:** Real implementations use RoPE in the indexer and specialized kernels (FlashMLA) for efficiency.

#### DSA-Math

$$ Q_{\text{idx}}, K_{\text{idx}} = \text{Proj}_{\text{light}}(X) $$
$$ \text{Scores} = \text{RoPE}(Q_{\text{idx}}) \cdot \text{RoPE}(K_{\text{idx}})^\top $$
$$ \mathcal{I} = \text{TopK}(\text{Scores}, k) $$
$$ K_{\text{sparse}} = \text{Gather}(K_{\text{full}}, \mathcal{I}) $$
$$ \text{Attn} = \text{Softmax}\left(\frac{Q K_{\text{sparse}}^\top}{\sqrt{d}}\right) V_{\text{sparse}} $$

#### DSA-Code

```python
class DeepSeekSparseAttention(nn.Module):
    """
    Corrected DSA: Query-Dependent Lightning Indexer
    Ref: DeepSeek-V3.2-Exp Technical Report
    """
    def __init__(self, dim: int, num_heads: int, k_sparse: int):
        super().__init__()
        self.k_sparse = k_sparse
        self.num_heads = num_heads
        
        # Main Attention Projections (Full Precision)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # Lightning Indexer (Compressed, e.g., dim/4)
        # In practice: uses FP8 and specialized RoPE layout
        self.idx_dim = dim // 4
        self.idx_q = nn.Linear(dim, self.idx_dim)
        self.idx_k = nn.Linear(dim, self.idx_dim)

    def apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotary Positional Embedding (Placeholder)
        Crucial for relative position awareness in the indexer.
        """
        # In production, use optimized CUDA kernels
        return x 
        
    def forward(self, x: torch.Tensor):
        B, N, D = x.shape
        
        # 1. Lightning Indexer Step
        # Project to compressed dimension
        q_idx = self.idx_q(x) # (B, N, D_idx)
        k_idx = self.idx_k(x) 
        
        # Apply RoPE (Crucial for long-context retrieval)
        q_idx = self.apply_rope(q_idx)
        k_idx = self.apply_rope(k_idx)
        
        # Compute scores (B, N, N) - Query vs All Keys
        scores = torch.bmm(q_idx, k_idx.transpose(1, 2))
        
        # 2. Top-K Selection
        k = min(self.k_sparse, N)
        _, top_indices = torch.topk(scores, k, dim=-1) # (B, N, k)
        
        # 3. Sparse Attention (FlashMLA Integration)
        # In a real setting, we invoke the FlashMLA kernel here to avoid
        # materializing the full attention matrix or full KV cache.
        
        # >>> FlashMLA.sparse_attention(q, k, v, top_indices) <<<
        
        # Fallback / Conceptual implementation:
        Q = self.q_proj(x).view(B, N, self.num_heads, -1)
        K = self.k_proj(x).view(B, N, self.num_heads, -1)
        V = self.v_proj(x).view(B, N, self.num_heads, -1)
        
        # Gather K/V using top_indices...
        # K_sparse = gather(K, top_indices)
        # V_sparse = gather(V, top_indices)
        
        # 4. Sparse Attention
        # attn = softmax(Q @ K_sparse.T) @ V_sparse
        
        return x # Placeholder return
```

### 🚀 Hybrid Integration Options

#### Option 1: Sequential Stack (KDA → DSA)

**Logic:** First, compress the entire history into a compact state using KDA (Linear). Then, use DSA to retrieve specific details from the *processed* sequence or a short local window.

```python
class SequentialHybrid(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.kda = KimiDeltaAttention(dim)
        self.dsa = DeepSeekSparseAttention(dim, k_sparse=64)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, state=None):
        # 1. Global Context Compression (Linear)
        res = x
        x, state = self.kda(x, state=state)
        x = self.norm(x + res)
        
        # 2. Local/Sparse Refinement (Sparse)
        res = x
        x = self.dsa(x)
        x = self.norm(x + res)
        
        return x, state
```

#### Option 2: Parallel Merge with Gating

**Logic:** Run both mechanisms in parallel. A learned gate decides for each token whether to rely on the global summary (KDA) or specific retrieved tokens (DSA).

```python
class ParallelHybrid(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.kda = KimiDeltaAttention(dim)
        self.dsa = DeepSeekSparseAttention(dim)
        self.gate_net = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, state=None):
        # Parallel Execution
        out_kda, state = self.kda(x, state=state)
        out_dsa = self.dsa(x)
        
        # Learned Fusion
        alpha = self.gate_net(x) # (B, N, 1)
        
        # Blend: alpha * KDA + (1-alpha) * DSA
        out = alpha * out_kda + (1 - alpha) * out_dsa
        return out, state
```

#### Option 3: Layer-wise Interleaving (3:1 Ratio)

**Logic:** As recommended in the Kimi Linear paper, interleave KDA layers with DSA/Attention layers. A 3:1 ratio (3 KDA : 1 DSA) provides optimal balance between efficiency and precision.

```python
class LayerWiseHybridModel(nn.Module):
    def __init__(self, dim, depth=12):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(depth):
            # Every 4th layer is DSA (Refinement), others KDA (Throughput)
            if (i + 1) % 4 == 0:
                layer = DeepSeekSparseAttention(dim, k_sparse=64)
            else:
                layer = KimiDeltaAttention(dim)
            self.layers.append(layer)

    def forward(self, x, state=None):
        # Note: State management becomes complex with interleaving
        # Usually KDA layers pass state, DSA layers ignore it
        for layer in self.layers:
            if isinstance(layer, KimiDeltaAttention):
                x, state = layer(x, state)
            else:
                x = layer(x)
        return x, state
```

### 📁 Recommended Codebase Structure

```text
hybrid_attention/
├── attention/
│   ├── __init__.py
│   ├── kda.py          # KimiDeltaAttention (DPLR)
│   ├── dsa.py          # DeepSeekSparseAttention (FlashMLA)
│   ├── hybrid.py       # Fusion modules
│   └── kernels.py      # Triton kernels for chunkwise/sparse ops
├── requirements.txt
└── README.md
```
