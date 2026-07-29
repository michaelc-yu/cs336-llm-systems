import torch
import math
from einops import einsum, rearrange
import triton
import triton.language as tl


# Tile sizes
B_Q = 16
B_K = 16

def flashattention_bwd(Q, K, V, O, dO, L, is_causal=False):
    """
    Inputs: Q, K, V, O, dO, L
    Outputs: dQ, dK, dV
    """
    batch_sz, N_q, d = Q.shape
    _, N_k, _ = K.shape

    S = einsum(Q, K, "b q d, b k d -> b q k") / math.sqrt(d)

    if is_causal:
        mask = torch.tril(torch.ones(N_q, N_k, device=Q.device, dtype=Q.dtype))
        S = S.masked_fill(mask == 0, -float('inf'))

    # S is shape (b, q, k)
    # L is shape (b, q)
    P_ij = torch.exp(S - rearrange(L, "b q -> b q 1"))

    dV = einsum(P_ij, dO, "b q k, b q d -> b k d")
    dP = einsum(dO, V, "b q d, b v d -> b q v")

    D = torch.sum(O * dO, dim=-1)
    dS = P_ij * (dP - rearrange(D, "b q -> b q 1"))

    dQ = einsum(dS, K, "b q k, b k d -> b q d") / math.sqrt(d)
    dK = einsum(dS, Q, "b q k, b q d -> b k d") / math.sqrt(d)
    return dQ, dK, dV

bwd_compiled = torch.compile(flashattention_bwd)


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Inputs: Q, K, V
        Outputs: O, L
        """
        # For one batch:
        # Q has shape Nq x d
        # K has shape Nk x d
        # V has shape Nk x d
        assert K.shape[1] == V.shape[1]

        batch_sz, N_q, d = Q.shape
        _, N_k , _ = K.shape
        
        assert N_q % B_Q == 0 # num rows in Q is divisible by tile size
        assert N_k % B_K == 0 # num rows in K,V is divisible by tile size

        T_q = math.ceil(N_q / B_Q)
        T_k = math.ceil(N_k / B_K)

        device = Q.device
        dtype = Q.dtype

        O = torch.zeros((batch_sz, N_q, d), device=device, dtype=dtype)
        L = torch.zeros((batch_sz, N_q), device=device, dtype=dtype) # logsumexp

        for b in range(batch_sz):
            for i in range(T_q):
                # Load Q_i from global memory
                q_start = i * B_Q
                q_end = q_start + B_Q
                Q_i = Q[b][q_start : q_end :]
                # Initialize O_i and L_i
                O_ij = O[b][q_start : q_end :]
                L_ij = L[b][q_start : q_end]
                m_ij = torch.full((B_Q,), float('-inf'), device=device, dtype=dtype)

                scale = 1 / math.sqrt(d)

                for j in range(T_k):
                    # Load K_j and V_j from global memory
                    k_start = j * B_K
                    k_end = k_start + B_K
                    K_j = K[b][k_start : k_end :]
                    V_j = V[b][k_start : k_end :]

                    S_ij = einsum(Q_i, K_j, "B_q d, B_k d -> B_q B_k") * scale

                    if is_causal:
                        query_indices = i * B_Q + torch.arange(B_Q, device=Q.device)
                        key_indices = j * B_K + torch.arange(B_K, device=Q.device)
                        mask = query_indices[:, None] >= key_indices[None, :]
                        S_ij = torch.where(mask, S_ij, S_ij - 1e6)

                    prev_mij = m_ij
                    m_ij = torch.max(m_ij, torch.max(S_ij, dim=-1).values)

                    # S_ij has shape B_q, B_k
                    # m_ij has shape B_q,
                    P_ij = torch.exp(S_ij - rearrange(m_ij, "(B_q x) -> B_q x", x=1))

                    L_ij = torch.exp(prev_mij - m_ij) * L_ij + torch.sum(P_ij, dim=-1)
                    O_ij = einsum(torch.diag(torch.exp(prev_mij - m_ij)), O_ij, "B_q B_q, B_q d -> B_q d") + einsum(P_ij, V_j, "B_q B_k, B_k d -> B_q d")
                
                # Compute Oi and Li
                O_ij = einsum(torch.diag(1 / L_ij), O_ij, "B_q B_q, B_q d -> B_q d")
                L_ij = m_ij + torch.log(L_ij)

                # Write Oi to global memory as the i-th tile of O
                O[b][q_start : q_end :] = O_ij
                # Write Li to global memory as the i-th tile of L
                L[b][q_start : q_end] = L_ij

        ctx.is_causal = is_causal
        ctx.save_for_backward(L, Q, K, V, O)

        return O

    @staticmethod
    def backward(ctx, dO):
        """
        Inputs: Q, K, V, O, dO, L
        Outputs: dQ, dK, dV
        """
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = bwd_compiled(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    # This is equivalent to what we did above with Qi = Q[b][i*Bq:(i+1)*Bq]
    # just getting the ptr to the right tensor slice before loading it
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Load Q_i from global memory
    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1))

    # Initialize O_i, l_i, m_i
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)

    T_k = tl.cdiv(N_KEYS, K_TILE_SIZE)

    for j in range(T_k):
        # Load K_j and V_j from global memory
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1))
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1))

        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale

        if is_causal:
            query_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            key_indices = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = tl.reshape(query_indices, (Q_TILE_SIZE, 1)) >= tl.reshape(key_indices, (1, K_TILE_SIZE))
            S_ij = tl.where(mask, S_ij, S_ij - 1e6)

        m_ij = tl.maximum(m_i, tl.max(S_ij, axis = -1))
        P_ij = tl.exp(S_ij - tl.reshape(m_ij, (Q_TILE_SIZE, 1)))

        l_ij = tl.exp(m_i - m_ij) * l_i + tl.sum(P_ij, axis=-1)
        O_ij = tl.reshape(tl.exp(m_i - m_ij), (Q_TILE_SIZE, 1)) * O_i + tl.dot(P_ij.to(V_j.dtype), V_j)

        # Update l_i, O_i, m_i
        l_i = l_ij
        O_i = O_ij
        m_i = m_ij

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    O_i_final = tl.reshape(1 / l_i, (Q_TILE_SIZE, 1)) * O_i
    l_i_final = m_i + tl.log(l_i)

    O_i_final = O_i_final.to(Q_i.dtype)
    l_i_final = l_i_final.to(Q_i.dtype)

    tl.store(O_block_ptr, O_i_final, boundary_check=(0, 1))
    tl.store(L_block_ptr, l_i_final, boundary_check=(0,))


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # For one batch:
        # Q has shape Nq x d
        # K has shape Nk x d
        # V has shape Nk x d
        assert K.shape[1] == V.shape[1]

        batch_sz, N_q, d = Q.shape
        _, N_k , _ = K.shape
        
        assert N_q % B_Q == 0 # num rows in Q is divisible by tile size
        assert N_k % B_K == 0 # num rows in K,V is divisible by tile size

        T_q = math.ceil(N_q / B_Q)
        T_k = math.ceil(N_k / B_K)

        device = Q.device
        dtype = Q.dtype

        O = torch.empty((batch_sz, N_q, d), device=device, dtype=dtype)
        L = torch.empty((batch_sz, N_q), device=device, dtype=dtype) # logsumexp

        ctx.is_causal = is_causal
        ctx.Q_TILE_SIZE = B_Q
        ctx.K_TILE_SIZE = B_K

        scale = 1 / math.sqrt(d)
        flash_fwd_kernel[(math.ceil(N_q / ctx.Q_TILE_SIZE), batch_sz)](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_q, N_k,
            scale,
            d,
            B_Q,
            B_K,
            is_causal=ctx.is_causal
        )

        ctx.save_for_backward(Q, K, V, O, L)
        
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = bwd_compiled(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None

