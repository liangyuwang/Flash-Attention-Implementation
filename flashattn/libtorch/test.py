import time
from tqdm.auto import tqdm
import torch
from torch.cuda import memory_allocated
from torch.utils.cpp_extension import load

# Compiling and loading the C++ module
flashattn = load(
    name='flashattn',
    sources=['./flashattn.cpp'],
    extra_cflags=['-O2'],
    verbose=True
)

def attention(query, key, value, scale=None, mask=None, if_causal=True):
    attention_scores = torch.matmul(query, key.transpose(-2, -1))

    scale = scale if scale is not None else key.shape[-1] ** -0.5
    attention_scores = attention_scores * scale
    
    if if_causal and mask is None:
        causal_mask = torch.tril(torch.ones(attention_scores.shape[-2:], device=attention_scores.device))
        attention_scores = attention_scores.masked_fill(causal_mask==0, float('-inf'))
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask==0, float('-inf'))
    
    attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


def eval_acc():
    # Setup
    torch.manual_seed(0)
    BS, dim, heads, seq_len = 2, 4096, 8, 1024
    Br, Bc = 32, 64
    if_causal = True
    df = torch.float32
    # Test input
    q = torch.randn(BS, heads, seq_len, dim, requires_grad=True).to(df).cuda()
    k = torch.randn(BS, heads, seq_len, dim, requires_grad=True).to(df).cuda()
    v = torch.randn(BS, heads, seq_len, dim, requires_grad=True).to(df).cuda()
    dOut = torch.randn_like(q)

    # Forward pass
    out_standard, _ = attention(q, k, v, if_causal=if_causal)
    out_flash, logsumexp = flashattn.forward(q, k, v, Br, Bc, if_causal=if_causal)

    # Calculate absolute differences
    abs_diff = torch.abs(out_standard - out_flash).max().item()
    print(f'Max absolute difference between standard and flash attention forward outputs: {abs_diff:.15f}')

    # Backward pass
    dQ_standard, dK_standard, dV_standard = torch.autograd.grad(outputs=out_standard, inputs=[q, k, v], grad_outputs=dOut, create_graph=True)
    dQ_flash, dK_flash, dV_flash = flashattn.backward(dOut, q, k, v, out_flash, logsumexp, Br, Bc, if_causal=if_causal)

    # Calculate and print differences between the gradients
    diff_dQ = torch.abs(dQ_standard - dQ_flash).max().item()
    diff_dK = torch.abs(dK_standard - dK_flash).max().item()
    diff_dV = torch.abs(dV_standard - dV_flash).max().item()

    print(f"Max absolute differences in gradients between standard and flash attention:")
    print(f"dQ difference: {diff_dQ:.15f}")
    print(f"dK difference: {diff_dK:.15f}")
    print(f"dV difference: {diff_dV:.15f}")

    # Print out gradient magnitudes to compare
    print("Standard attention gradients:")
    print("dQ:", dQ_standard.norm().item(), "dK:", dK_standard.norm().item(), "dV:", dV_standard.norm().item())
    print("Flash attention gradients:")
    print("dQ:", dQ_flash.norm().item(), "dK:", dK_flash.norm().item(), "dV:", dV_flash.norm().item())

def eval_speed():
    torch.manual_seed(0)
    def _calculate_time(fn, *args, **kwargs):
        # Warm-up 
        for _ in tqdm(range(2)):
            output = fn(*args, **kwargs)
        # Timed run
        start_time = time.time()
        for _ in range(1):
            _ = fn(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return elapsed_time, output

    BS, dim, heads, seq_len = 1, 1024, 8, 1024
    Br, Bc = 32, 64
    if_causal = True
    df = torch.float32
    # Test input
    q = torch.randn(BS, heads, seq_len, dim, requires_grad=True).to(df).cuda()
    k = torch.randn(BS, heads, seq_len, dim, requires_grad=True).to(df).cuda()
    v = torch.randn(BS, heads, seq_len, dim, requires_grad=True).to(df).cuda()
    dOut = torch.randn_like(q)

    # flash-attn forward
    flash_elapsed_time, (out_flash, logsumexp) = _calculate_time(flashattn.forward, q, k, v, Br, Bc, if_causal=if_causal)
    torch.cuda.empty_cache()
    standard_elapsed_time, (out_standard, _) = _calculate_time(attention, q, k, v, if_causal=if_causal)
    torch.cuda.empty_cache()
    print(f'flash-attn fwd / standard-attn fwd speed: {standard_elapsed_time/flash_elapsed_time:.4f}')

    # flash-attn backward
    flash_elapsed_time, _ = _calculate_time(flashattn.backward, dOut, q, k, v, out_flash, logsumexp, Br, Bc, if_causal=if_causal)
    torch.cuda.empty_cache()
    standard_elapsed_time, _ = _calculate_time(torch.autograd.grad, outputs=out_standard, inputs=[q, k, v], grad_outputs=dOut, create_graph=True)
    torch.cuda.empty_cache()
    print(f'flash-attn bwd / standard-attn bwd speed: {standard_elapsed_time/flash_elapsed_time:.4f}')

def eval_memory():
    torch.manual_seed(0)
    def _calculate_memory(fn, *args, **kwargs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        memory_before = memory_allocated()
        output = fn(*args, **kwargs)
        memory_after = memory_allocated()

        peak_memory = torch.cuda.max_memory_allocated()
        memory_used = memory_after - memory_before
        return peak_memory, memory_used, output
    
    BS, dim, heads, seq_len = 1, 4096, 8, 1024
    Br, Bc = 32, 64
    if_causal = True
    df = torch.float32
    # Test input
    q = torch.randn(BS, heads, seq_len, dim, requires_grad=True).to(df).cuda()
    k = torch.randn(BS, heads, seq_len, dim, requires_grad=True).to(df).cuda()
    v = torch.randn(BS, heads, seq_len, dim, requires_grad=True).to(df).cuda()
    dOut = torch.randn_like(q)
    
    flash_peak_memory, flash_memory_used, (out_flash, logsumexp) = _calculate_memory(flashattn.forward, q, k, v, Br, Bc, if_causal=if_causal)
    standard_peak_memory, standard_memory_used, (out_standard, _) = _calculate_memory(attention, q, k, v, if_causal=if_causal)
    print(f'flash-attn fwd / standard-attn fwd memory usage: {flash_peak_memory / standard_peak_memory:.4f}')
    
    # flash-attn backward
    flash_peak_memory, flash_memory_used, _ = _calculate_memory(flashattn.backward, dOut, q, k, v, out_flash, logsumexp, Br, Bc, if_causal=if_causal)
    standard_peak_memory, standard_memory_used, _ = _calculate_memory(torch.autograd.grad, outputs=out_standard, inputs=[q, k, v], grad_outputs=dOut, create_graph=False)
    print(f'flash-attn bwd / standard-attn bwd memory usage: {flash_peak_memory / standard_peak_memory:.4f}')


if __name__=="__main__":
    eval_memory()
    torch.cuda.empty_cache()
    eval_acc()
    torch.cuda.empty_cache()
    eval_speed()
