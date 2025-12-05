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

# 🔬 Hybrid Attention Mechanisms: Kimi Linear & DeepSeek Sparse

> [!summary] Executive Summary
> This note explores the integration of **Kimi Delta Attention (KDA)** and **DeepSeek Sparse Attention (DSA)** to create efficient, high-performance Transformer architectures.
>
> * **KDA** (from [Kimi Linear](https://arxiv.org/abs/2510.26692)) offers $O(N)$ efficiency with fine-grained forgetting.
> * **DSA** (from [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp)) enables precise long-context retrieval via sparse access.

---

## 1. Kimi Delta Attention (KDA)

**Source:** *Kimi Linear: An Expressive, Efficient Attention Architecture* (arXiv:2510.26692)

> [!info] At a Glance
>
> * **Complexity:** Linear $O(N)$ time & memory.
> * **Core Innovation:** Channel-wise gated DeltaNet with DPLR-style decay.
> * **Best For:** Efficiently compressing global context into a fixed-size state.

### Mechanism

1. **Feature Projection:** Input $X$ is projected to Query ($Q$), Key ($K$), Value ($V$), and a special **Forget Gate** ($\beta$).
2. **Data-Dependent Decay:** Unlike standard Linear Attention, KDA computes a **channel-wise decay rate** $\mathbf{g}_t$ based on the input token. This allows the model to selectively "forget" irrelevant information per feature dimension.
3. **Chunkwise Computation:** Tokens are processed in chunks (e.g., 64) to leverage GPU Tensor Cores, balancing the sequential nature of RNNs with the parallel efficiency of Transformers.

### Mathematical Formulation

$$
\begin{aligned}
\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t, \mathbf{\beta}_t &= \text{Proj}(\mathbf{x}_t) \\
\mathbf{g}_t &= \sigma(\mathbf{\beta}_t) \quad \text{(Data-Dependent Decay)} \\
\mathbf{S}_t &= \mathbf{S}_{t-1} \odot \mathbf{g}_t + \mathbf{k}_t^\top \mathbf{v}_t \\
\mathbf{o}_t &= \text{LayerNorm}(\mathbf{q}_t \mathbf{S}_t)
\end{aligned}
$$

> [!tip] Note on $\mathbf{g}_t$
> The decay rate $\mathbf{g}_t$ is **broadcasted** across the $D \times D$ state matrix. This channel-wise gating is what differentiates KDA from simple DeltaNet or RWKV, allowing for more expressive memory management.

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class KimiDeltaAttention(nn.Module):
    """
    KDA: Gated DeltaNet with Data-Dependent Decay.
    Based on Kimi Linear (arXiv:2510.26692).
    """
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Projections: Q, K, V, and Beta (Decay/Gate)
        self.qkv_gate = nn.Linear(dim, 3 * dim + dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor, state: torch.Tensor = None):
        B, N, D = x.shape
        H, HD = self.num_heads, self.head_dim
        
        # 1. Project inputs
        proj = self.qkv_gate(x)
        q, k, v, beta = torch.split(proj, [D, D, D, D], dim=-1)
        
        # 2. Reshape & Activation
        q, k, v = [t.view(B, N, H, HD) for t in (q, k, v)]
        g = torch.sigmoid(beta.view(B, N, H, HD)) # Decay rate in (0, 1)
        
        # 3. Initialize State (B, H, D, D)
        if state is None:
            state = torch.zeros(B, H, HD, HD, device=x.device)
            
        outputs = []
        
        # 4. Recurrent Update (Simplified; real impl uses Chunkwise DPLR)
        for t in range(N):
            q_t = q[:, t]
            k_t = k[:, t]
            v_t = v[:, t]
            g_t = g[:, t].unsqueeze(-1) # Broadcast over last dim
            
            # S_t = S_{t-1} * g_t + K^T * V
            state = state * g_t + torch.einsum('bhd,bhm->bhdm', k_t, v_t)
            
            # O_t = Q_t * S_t
            out_t = torch.einsum('bhd,bhdm->bhm', q_t, state)
            outputs.append(out_t)
            
        output = torch.stack(outputs, dim=1).reshape(B, N, D)
        return self.out_proj(output), state
```

---

## 2. DeepSeek Sparse Attention (DSA)

**Source:** *DeepSeek-V3.2-Exp Technical Report*

> [!info] At a Glance
>
> * **Complexity:** Sparse access (Top-K).
> * **Core Innovation:** Query-Dependent "Lightning Indexer" + FlashMLA kernels.
> * **Best For:** Retrieving specific "needle-in-a-haystack" details from massive contexts.

### Mechanism

1. **Lightning Indexer:** A lightweight, compressed attention branch (often FP8) is used to quickly estimate token relevance.
2. **Query-Dependent Scoring:** Unlike static sparse methods, DSA computes relevance scores dynamically between the *current query* and *all past compressed keys*.
3. **Top-K Selection:** The indices of the top-$k$ most relevant tokens are selected.
4. **Sparse Gather:** Full-precision KV pairs are fetched only for these selected indices.

> [!warning] Implementation Detail
> Real-world DSA relies heavily on **RoPE (Rotary Positional Embeddings)** in the indexer to maintain relative position awareness, and specialized **FlashMLA** CUDA kernels to avoid materializing full attention matrices.

### Mathematical Formulation

$$
\begin{aligned}
Q_{\text{idx}}, K_{\text{idx}} &= \text{Proj}_{\text{light}}(X) \\
\text{Scores} &= \text{RoPE}(Q_{\text{idx}}) \cdot \text{RoPE}(K_{\text{idx}})^\top \\
\mathcal{I} &= \text{TopK}(\text{Scores}, k) \\
K_{\text{sparse}} &= \text{Gather}(K_{\text{full}}, \mathcal{I}) \\
\text{Attn} &= \text{Softmax}\left(\frac{Q K_{\text{sparse}}^\top}{\sqrt{d}}\right) V_{\text{sparse}}
\end{aligned}
$$

### PyTorch Implementation (Conceptual)

```python
class DeepSeekSparseAttention(nn.Module):
    """
    DSA: Query-Dependent Lightning Indexer + Sparse Gather.
    Ref: DeepSeek-V3.2-Exp
    """
    def __init__(self, dim: int, num_heads: int, k_sparse: int = 64):
        super().__init__()
        self.k_sparse = k_sparse
        self.num_heads = num_heads
        
        # Full Precision Projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # Lightning Indexer (Compressed)
        self.idx_dim = dim // 4
        self.idx_q = nn.Linear(dim, self.idx_dim)
        self.idx_k = nn.Linear(dim, self.idx_dim)

    def forward(self, x: torch.Tensor):
        B, N, D = x.shape
        
        # 1. Lightning Indexer (Score Computation)
        q_idx = self.idx_q(x) 
        k_idx = self.idx_k(x)
        
        # (Apply RoPE here in real impl)
        
        scores = torch.bmm(q_idx, k_idx.transpose(1, 2))
        
        # 2. Top-K Selection
        k = min(self.k_sparse, N)
        _, top_indices = torch.topk(scores, k, dim=-1) 
        
        # 3. Sparse Attention (Simulated)
        # In practice: call FlashMLA.sparse_attention(q, k, v, top_indices)
        
        return x # Placeholder for full sparse op
```

---

## 3. Hybrid Integration Strategies

Combining KDA's global compression with DSA's precise retrieval offers a "best of both worlds" architecture.

### Option A: Sequential Stack (Compress → Refine)

**Logic:** Use KDA layers to maintain a running summary of the context, followed by DSA layers to "zoom in" on specific details when needed.

```python
class SequentialHybrid(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.kda = KimiDeltaAttention(dim)
        self.dsa = DeepSeekSparseAttention(dim, k_sparse=64)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, state=None):
        res = x
        # Global Compression
        x, state = self.kda(x, state=state)
        x = self.norm(x + res)
        
        res = x
        # Local Refinement
        x = self.dsa(x)
        x = self.norm(x + res)
        return x, state
```

### Option B: Parallel Gated Merge

**Logic:** Run both branches in parallel and let the model learn when to rely on memory (KDA) vs. retrieval (DSA).

```python
class ParallelHybrid(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.kda = KimiDeltaAttention(dim)
        self.dsa = DeepSeekSparseAttention(dim)
        self.gate = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, x, state=None):
        out_kda, state = self.kda(x, state=state)
        out_dsa = self.dsa(x)
        
        alpha = self.gate(x) # Dynamic weighting
        return alpha * out_kda + (1 - alpha) * out_dsa, state
```

### Option C: Layer-wise Interleaving (Recommended)

**Logic:** Interleave layers in a fixed ratio (e.g., 3 KDA : 1 DSA). This is the approach suggested by the Kimi Linear paper for optimal performance/efficiency balance.

```python
class LayerWiseHybrid(nn.Module):
    def __init__(self, dim, depth=12):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(depth):
            if (i + 1) % 4 == 0:
                self.layers.append(DeepSeekSparseAttention(dim, k_sparse=64))
            else:
                self.layers.append(KimiDeltaAttention(dim))

    def forward(self, x, state=None):
        for layer in self.layers:
            if isinstance(layer, KimiDeltaAttention):
                x, state = layer(x, state)
            else:
                x = layer(x)
        return x, state
```

---

## 📁 Project Structure

Recommended folder structure for a hybrid implementation:

```text
hybrid_attention/
├── attention/
│   ├── __init__.py
│   ├── kda.py          # KimiDeltaAttention (DPLR)
│   ├── dsa.py          # DeepSeekSparseAttention (FlashMLA)
│   ├── hybrid.py       # Fusion modules
│   └── kernels.py      # Triton/CUDA kernels
├── requirements.txt
└── README.md
```
