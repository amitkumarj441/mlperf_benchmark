from typing import Optional, Tuple

import torch
import torch.nn as nn
import triton
import triton.language as tl
from transformers.models.llama.modeling_llama import (LlamaAttention,
                                                      apply_rotary_pos_emb,
                                                      repeat_kv)
from transformers.utils import logging

logger = logging.get_logger(__name__)


@triton.jit
def abc_fwd_kernel(
    q, k, v, o, sk, sv,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_skb, stride_skh, stride_skt, stride_skd,
    B, H, T, D, M, scale,
    BD: tl.constexpr,
    BM: tl.constexpr
):
    i_bh = tl.program_id(0)
    p_q = tl.make_block_ptr(base=q + i_bh * stride_qh,
                            shape=(T * D,),
                            strides=(stride_qd,),
                            offsets=(0,),
                            block_shape=(BD,),
                            order=(0,))
    p_k = tl.make_block_ptr(base=k + i_bh * stride_qh,
                            shape=(T * D,),
                            strides=(stride_qd,),
                            offsets=(0,),
                            block_shape=(BD,),
                            order=(0,))
    p_v = tl.make_block_ptr(base=v + i_bh * stride_qh,
                            shape=(T * D,),
                            strides=(stride_qd,),
                            offsets=(0,),
                            block_shape=(BD,),
                            order=(0,))
    p_o = tl.make_block_ptr(base=o + i_bh * stride_qh,
                            shape=(T * D,),
                            strides=(stride_qd,),
                            offsets=(0,),
                            block_shape=(BD,),
                            order=(0,))
    p_sk = tl.make_block_ptr(base=sk + i_bh * stride_skh,
                             shape=(T * M,),
                             strides=(stride_skd,),
                             offsets=(0,),
                             block_shape=(BM,),
                             order=(0,))
    p_sv = tl.make_block_ptr(base=sv + i_bh * stride_skh,
                             shape=(T * M,),
                             strides=(stride_skd,),
                             offsets=(0,),
                             block_shape=(BM,),
                             order=(0,))
    m_sk, m_sv = tl.full([BM,], float('-inf'), dtype=tl.float32), tl.full([BM,], float('-inf'), dtype=tl.float32)
    a_sk, a_sv = tl.zeros([BM,], dtype=tl.float32), tl.zeros([BM,], dtype=tl.float32)
    a_k = tl.zeros([BM, BD], dtype=tl.float32)
    a_v = tl.zeros([BM, BD], dtype=tl.float32)

    for _ in range(T):
        # [BM,]
        b_sk = tl.load(p_sk)
        m_ski = tl.maximum(m_sk, b_sk)
        b_sk = tl.exp(b_sk - m_ski)
        a_sk = a_sk * tl.exp(m_sk - m_ski)
        a_ski = b_sk + a_sk
        # [BM, BD]
        a_k = a_k * (a_sk / a_ski)[:, None] + (b_sk / a_ski)[:, None] * tl.load(p_k)[None, :]

        # [BM,]
        b_sv = tl.load(p_sv)
        m_svi = tl.maximum(m_sv, b_sv)
        b_sv = tl.exp(b_sv - m_svi)
        a_sv = a_sv * tl.exp(m_sv - m_svi)
        a_svi = b_sv + a_sv
        # [BM, BD]
        a_v = a_v * (a_sv / a_svi)[:, None] + (b_sv / a_svi)[:, None] * tl.load(p_v)[None, :]

        # [BD,]
        b_q = tl.load(p_q) * scale
        # [BD,]
        b_o = tl.sum(tl.softmax(tl.sum(b_q[None, :] * a_k, 1), 0)[:, None] * a_v, 0)
        tl.store(p_o, b_o.to(p_q.dtype.element_ty))

        m_sk, m_sv = m_ski, m_svi
        a_sk, a_sv = a_ski, a_svi

        p_q = tl.advance(p_q, (BD,))
        p_k = tl.advance(p_k, (BD,))
        p_v = tl.advance(p_v, (BD,))
        p_o = tl.advance(p_o, (BD,))
        p_sk = tl.advance(p_sk, (BM,))
        p_sv = tl.advance(p_sv, (BM,))


class ABCAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sk, sv):
        BD, BM = q.shape[-1], sk.shape[-1]
        batch_size, n_heads, seq_len, d_head = q.shape
        num_stages = 3 if d_head <= 64 else 2
        num_warps = 4
        grid = (batch_size * n_heads,)
        scale = d_head ** -0.5
        assert d_head in {16, 32, 64, 128}

        o = torch.empty_like(q)
        abc_fwd_kernel[grid](
            q, k, v, o, sk, sv,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            sk.stride(0), sk.stride(1), sk.stride(2), sk.stride(3),
            batch_size, n_heads, seq_len, d_head, sk.shape[-1], scale,
            BD=BD, BM=BM,
            num_warps=num_warps,
            num_stages=num_stages
        )

        ctx.save_for_backward(q, k, v, sk, sv, o)
        ctx.grid = grid
        ctx.scale = scale
        return o

    @staticmethod
    def backward(ctx, do):
        def reversed_cumsum(x, dim=-1):
            c = x.cumsum(dim)
            return x + c.index_select(dim, x.new_tensor([c.shape[dim]-1], dtype=torch.long)) - c

        q, k, v, ek, ev, ak, av, p, o = ctx.saved_tensors
        scale = ctx.scale
        K = (ek.unsqueeze(-1) * k.unsqueeze(-2)).cumsum(2) / ak.unsqueeze(-1)
        V = (ev.unsqueeze(-1) * v.unsqueeze(-2)).cumsum(2) / av.unsqueeze(-1)

        dq, dk, dv, dsk, dsv = None, None, None, None, None
        dp = (p * (torch.einsum('...qd,...qmd->...qm', do, V) - (do * o).sum(-1, True))) * scale
        dq = torch.einsum('...qm,...qmd->...qd', dp, K)

        dK = torch.einsum('...qm,...qd->...qmd', dp / ak, q)
        dK1 = reversed_cumsum(dK, 2)
        dk = torch.einsum('...qm,...qmd->...qd', ek, dK1)
        dsk = ek * (torch.einsum('...qd,...qmd->...qm', k, dK1) - reversed_cumsum((dK * K).sum(-1), 2))

        dV = torch.einsum('...qd,...qm->...qmd', do, p / av)
        dV1 = reversed_cumsum(dV, 2)
        dv = torch.einsum('...qm,...qmd->...qd', ev, dV1)
        dsv = ev * (torch.einsum('...qd,...qmd->...qm', v, dV1) - reversed_cumsum((dV * V).sum(-1), 2))
        return dq, dk, dv, dsk, dsv


def naive_attention(q, k, v, sk, sv):
    dtype = q.dtype
    *_, d_head = q.shape
    # [batch_size, n_heads, seq_len, 64]
    ek = (sk - sk.max(2, True)[0]).exp()
    ev = (sv - sv.max(2, True)[0]).exp()
    ak, av = ek.cumsum(2), ev.cumsum(2)
    # [batch_size, n_heads, seq_len, 64, d_head]
    K = (ek.unsqueeze(-1) * k.unsqueeze(-2)).cumsum(2) / ak.unsqueeze(-1)
    V = (ev.unsqueeze(-1) * v.unsqueeze(-2)).cumsum(2) / av.unsqueeze(-1)
    # [batch_size, n_heads, seq_len, 64]
    p = torch.einsum('...d,...md->...m', q * d_head ** -0.5, K).softmax(-1, dtype=torch.float).to(dtype)
    # [batch_size, n_heads, seq_len, d_head]
    o = torch.einsum('...m,...md->...d', p, V)
    return o


class NaiveAttention1(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sk, sv):
        *_, d_head = q.shape
        dtype, scale = q.dtype, d_head ** -0.5
        # [batch_size, n_heads, seq_len, 64]
        ek = (sk - sk.max(2, True)[0]).to(torch.float).exp()
        ev = (sv - sv.max(2, True)[0]).to(torch.float).exp()
        ak, av = ek.cumsum(2), ev.cumsum(2)
        # [batch_size, n_heads, seq_len, 64, d_head]
        K = (ek.unsqueeze(-1) * k.unsqueeze(-2)).cumsum(2) / ak.unsqueeze(-1)
        V = ((ev.unsqueeze(-1) * v.unsqueeze(-2)).cumsum(2) / av.unsqueeze(-1)).to(dtype)
        # [batch_size, n_heads, seq_len, 64]
        p = torch.einsum('...d,...md->...m', q.to(torch.float) * scale, K).softmax(-1).to(dtype)
        # [batch_size, n_heads, seq_len, d_head]
        o = torch.einsum('...m,...md->...d', p, V)
        ctx.save_for_backward(q, k, v, ek, ev, ak, av, p, o)
        ctx.dtype, ctx.scale = dtype, scale
        return o

    @staticmethod
    def backward(ctx, do):
        def reversed_cumsum(x, dim=-1):
            c = x.cumsum(dim)
            return x + c.index_select(dim, x.new_tensor([c.shape[dim]-1], dtype=torch.long)) - c

        q, k, v, ek, ev, ak, av, p, o = ctx.saved_tensors
        dtype, scale = ctx.dtype, ctx.scale
        K = ((ek.unsqueeze(-1) * k.unsqueeze(-2)).cumsum(2) / ak.unsqueeze(-1)).to(dtype)
        V = ((ev.unsqueeze(-1) * v.unsqueeze(-2)).cumsum(2) / av.unsqueeze(-1)).to(dtype)

        dq, dk, dv, dsk, dsv = None, None, None, None, None
        dp = (p * (torch.einsum('...qd,...qmd->...qm', do, V) - (do * o).sum(-1, True))) * scale
        dq = torch.einsum('...qm,...qmd->...qd', dp, K)

        dK = torch.einsum('...qm,...qd->...qmd', (dp / ak).to(dtype), q)
        dK1 = reversed_cumsum(dK, 2)
        dk = torch.einsum('...qm,...qmd->...qd', ek.to(dtype), dK1)
        dsk = ek * (torch.einsum('...qd,...qmd->...qm', k, dK1) - reversed_cumsum((dK * K).sum(-1), 2))

        dV = torch.einsum('...qd,...qm->...qmd', do, (p / av).to(dtype))
        dV1 = reversed_cumsum(dV, 2)
        dv = torch.einsum('...qm,...qmd->...qd', ev.to(dtype), dV1)
        dsv = ev * (torch.einsum('...qd,...qmd->...qm', v, dV1) - reversed_cumsum((dV * V).sum(-1), 2))
        return dq, dk, dv, dsk, dsv


naive_attention1 = NaiveAttention1.apply

abc_attention = ABCAttention.apply


class LLaMAABCAttention(LlamaAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.w_k = nn.Linear(self.hidden_size, 64, bias=False)
        self.w_v = nn.Linear(self.hidden_size, 64, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LlamaFlashAttention2 attention does not support output_attentions
        output_attentions = False

        batch_size, seq_len, _ = hidden_states.shape
        # [batch_size, seq_len, n_heads * d_head]
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = k.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        if past_key_value is not None:  # reuse k, v, self_attention
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        past_key_value = (k, v) if use_cache else None

        # cast to half precision
        input_dtype = q.dtype
        if input_dtype == torch.float32:
            logger.warning_once("The input hidden states seems to be silently casted in float32.")
            q = q.to(self.config.torch_dtype)
            k = k.to(self.config.torch_dtype)
            v = v.to(self.config.torch_dtype)

        if getattr(self, "num_key_value_groups", None):
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

        # [batch_size, n_heads, seq_len, 64]
        sk = self.w_k(hidden_states).view(batch_size, 1, seq_len, -1).repeat(1, self.num_heads, 1, 1)
        sv = self.w_v(hidden_states).view(batch_size, 1, seq_len, -1).repeat(1, self.num_heads, 1, 1)

        o = naive_attention(q, k, v, sk, sv)
        o = o.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        o = self.o_proj(o)

        if not output_attentions:
            p = None

        return o, p, past_key_value


if __name__ == '__main__':
    B, H, T, D, M = 2, 8, 128, 32, 16
    dtype = torch.bfloat16
    torch.manual_seed(42)
    # [batch_size, n_heads, seq_len, d_head]
    q = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    # [batch_size, n_heads, seq_len, 64]
    sk = torch.randn((B, H, T, M), dtype=dtype, device='cuda').requires_grad_()
    sv = torch.randn((B, H, T, M), dtype=dtype, device='cuda').requires_grad_()

    do = torch.randn_like(q)
    ref = naive_attention(q, k, v, sk, sv)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dsk, sk.grad = sk.grad.clone(), None
    ref_dsv, sv.grad = sv.grad.clone(), None

    ref1 = naive_attention1(q, k, v, sk, sv)
    ref1.backward(do)
    ref1_dq, q.grad = q.grad.clone(), None
    ref1_dk, k.grad = k.grad.clone(), None
    ref1_dv, v.grad = v.grad.clone(), None
    ref1_dsk, sk.grad = sk.grad.clone(), None
    ref1_dsv, sv.grad = sv.grad.clone(), None
    #assert ref.allclose(ref1, 0, 1e-2)
    #import pdb
    #pdb.set_trace()
    #assert ref_dq.allclose(ref1_dq, 0, 1e-2)
    #assert ref_dk.allclose(ref1_dk, 0, 1e-2)
    #assert ref_dv.allclose(ref1_dv, 0, 1e-2)
    #assert ref_dsk.allclose(ref1_dsk, 0, 1e-2)
    #assert ref_dsv.allclose(ref1_dsv, 0, 1e-2)

    # triton implementation
    tri = abc_attention(q, k, v, sk, sv)
    # tri.backward(do)
    # tri_dv, v.grad = v.grad.clone(), None
    # tri_dk, k.grad = k.grad.clone(), None
    # tri_dq, q.grad = q.grad.clone(), None
    # assert ref.allclose(tri, 0, 1e-2)
    # assert torch.allclose(ref_dv, tri_dv, 0, 1e-2)
    # assert torch.allclose(ref_dk, tri_dk, 0, 1e-2)
    print('Done!')

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            # argument names to use as an x-axis for the plot
            x_names=['seq_len'],
            # different possible values for `x_name`
            x_vals=[128 * 2 ** i for i in range(0, 10)],
            # argument name whose value corresponds to a different line in the plot
            line_arg='provider',
            # possible values for `line_arg``
            line_vals=['torch', 'triton', 'torch_bwd', 'triton_bwd'],
            # label name for the lines
            line_names=['torch', 'triton', 'torch_bwd', 'triton_bwd'],
            # line styles
            styles=[('green', '-'), ('blue', '--'), ('red', '-.'), ('cyan', ':')],
            ylabel="Execution Time (ms)",  # label name for the y-axis
            # name for the plot. Used also as a file name for saving the plot.
            plot_name="Performance",
            args={},
        )
    )
    def benchmark(seq_len, provider):
        device = 'cuda'
        requires_grad = 'bwd' in provider
        batch_size, n_heads, d_head, n_mem = 2, 8, 64, 64

        q = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, requires_grad=requires_grad)
        k = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, requires_grad=requires_grad)
        v = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, requires_grad=requires_grad)
        sk = torch.randn(batch_size, n_heads, seq_len, n_mem, device=device, requires_grad=requires_grad)
        sv = torch.randn(batch_size, n_heads, seq_len, n_mem, device=device, requires_grad=requires_grad)
        do = torch.ones_like(q)

        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch':
            if seq_len > 40000:
                return 0, 0, 0
            results = triton.testing.do_bench(lambda: naive_attention(q, k, v, sk, sv), quantiles=quantiles)
        elif provider == 'triton':
            results = triton.testing.do_bench(lambda: abc_attention(q, k, v, sk, sv), quantiles=quantiles)
        elif provider == 'torch_bwd':
            if seq_len > 20000:
                return 0, 0, 0
            results = triton.testing.do_bench(lambda: naive_attention(q, k, v, sk, sv).backward(do), quantiles=quantiles)
        elif provider == 'triton_bwd':
            if seq_len > 20000:
                return 0, 0, 0
            results = triton.testing.do_bench(lambda: naive_attention1(q, k, v, sk, sv).backward(do), quantiles=quantiles)
        return results
    benchmark.run(show_plots=True, print_data=True, save_path='.')
