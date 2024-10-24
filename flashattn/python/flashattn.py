import torch


def flashattn_forward(
        Query: torch.Tensor, Key: torch.Tensor, Value: torch.Tensor, 
        Br: int, Bc: int, 
        scale: float=None, mask: torch.Tensor=None, if_causal: bool=True):
    """
    follow all steps and variables' names in https://tridao.me/publications/flash2/flash2.pdf Algorithm 1.
    'add scale', 'add mask': The only difference compared with Algorithm 1.
    """

    BS, n_qhead, N, dim = Query.size(0), Query.size(1), Query.size(2), Query.size(3)
    Out = torch.empty_like(Query)
    Logsumexp = torch.empty((BS, n_qhead, N), device=Query.device, dtype=Query.dtype)

    # add scale
    scale = scale if scale is not None else dim ** -0.5

    # add mask
    if if_causal and mask is None:
        causal_mask = torch.tril(torch.ones((N,N), device=Query.device, dtype=Query.dtype))
    
    for b in range(BS):
        for h in range(n_qhead):
            Q, K, V = Query[b,h,:,:], Key[b,h,:,:], Value[b,h,:,:]
            O = torch.empty_like(Q)
            L = torch.empty(Q.size(0), device=Q.device, dtype=Q.dtype)

            Q_Tr_blocks = torch.stack(Q.split(Br, dim=0), dim=0)
            O_Tr_blocks = torch.stack(O.split(Br, dim=0), dim=0)
            L_Tr_blocks = torch.stack(L.split(Br), dim=0)
            Tr = Q_Tr_blocks.size(0)

            K_Tc_blocks = torch.stack(K.split(Bc, dim=0), dim=0)
            V_Tc_blocks = torch.stack(V.split(Bc, dim=0), dim=0)
            Tc = K_Tc_blocks.size(0)

            for i in range(Tr):
                Qi = Q_Tr_blocks[i]
                Oi0 = O_Tr_blocks[i].fill_(0)
                li0 = torch.zeros(Br, device=Q.device, dtype=Q.dtype)
                mi0 = torch.empty_like(li0).fill_(-torch.inf)
                
                for j in range(Tc):
                    Kj, Vj = K_Tc_blocks[j], V_Tc_blocks[j]
                    Sij = torch.matmul(Qi, Kj.transpose(0, 1))

                    # add scale
                    Sij.mul_(scale)

                    # add mask
                    if if_causal:
                        causal_mask_ij = causal_mask[i*Br:(i+1)*Br, j*Bc:(j+1)*Bc]
                        if mask is None:
                            Sij = Sij.masked_fill(causal_mask_ij==0, float('-inf'))
                        else:
                            raise NotImplementedError
                    if mask is not None and not if_causal:
                        mask_ij = mask[i*Br:(i+1)*Br, j*Bc:(j+1)*Bc]
                        Sij = Sij.masked_fill(mask_ij==0, float('-inf'))

                    if j==0: mij_1 = mi0
                    Sij_rowmax, _ = torch.max(Sij, dim=-1)
                    mij = torch.max(mij_1, Sij_rowmax)
                    Pij = torch.exp(Sij - mij.unsqueeze(-1))
                    if j==0: lij_1 = li0
                    lij = torch.exp(mij_1 - mij) * lij_1 + torch.sum(Pij, dim=-1)
                    if j==0: Oij_1 = Oi0
                    Oij = torch.matmul(torch.diag(torch.exp(mij_1 - mij)), Oij_1) + torch.matmul(Pij, Vj)

                    mij_1, lij_1, Oij_1 = mij, lij, Oij
                li_Tc, Oi_Tc, mi_Tc, li_Tc = lij, Oij, mij, lij

                Oi = torch.matmul(torch.inverse(torch.diag(li_Tc)), Oi_Tc)
                Li = mi_Tc + torch.log(li_Tc)

                O_Tr_blocks[i,:,:], L_Tr_blocks[i,:] = Oi, Li
            Out[b,h,:,:] = O_Tr_blocks.reshape(-1, O_Tr_blocks.shape[-1])
            Logsumexp[b,h,:] = L_Tr_blocks.reshape(-1)
    
    return Out, Logsumexp


def flashattn_backward(
        dOut: torch.Tensor, 
        Query: torch.Tensor, Key: torch.Tensor, Value: torch.Tensor, 
        Out: torch.Tensor, Logsumexp: torch.Tensor,
        Br: int, Bc: int, 
        scale: float=None, mask: torch.Tensor=None, if_causal: bool=True):
    """
    follow all steps and variables' names in https://tridao.me/publications/flash2/flash2.pdf Algorithm 2.
    """

    BS, n_qhead, N, dim = Query.size(0), Query.size(1), Query.size(2), Query.size(3)
    dQuery, dKey, dValue = torch.empty_like(Query), torch.empty_like(Key), torch.empty_like(Value)
    
    # add scale
    scale = scale if scale is not None else dim ** -0.5

    for b in range(BS):
        for h in range(n_qhead):
            Q, K, V = Query[b,h,:,:], Key[b,h,:,:], Value[b,h,:,:]
            dQ, dK, dV = dQuery[b,h,:,:], dKey[b,h,:,:], dValue[b,h,:,:]
            O, L = Out[b,h,:,:], Logsumexp[b,h,:]
            dO = dOut[b,h,:,:]
            D = torch.sum(dO * O, dim=-1)

            Q_Tr_blocks = torch.stack(Q.split(Br, dim=0), dim=0)
            dQ_Tr_blocks = torch.stack(dQ.split(Br, dim=0), dim=0)
            O_Tr_blocks = torch.stack(O.split(Br, dim=0), dim=0)
            dO_Tr_blocks = torch.stack(dO.split(Br, dim=0), dim=0)
            L_Tr_blocks = torch.stack(L.split(Br), dim=0)
            D_Tr_blocks = torch.stack(D.split(Br), dim=0)
            Tr = dQ_Tr_blocks.size(0)

            K_Tc_blocks = torch.stack(K.split(Bc, dim=0), dim=0)
            V_Tc_blocks = torch.stack(V.split(Bc, dim=0), dim=0)
            dK_Tc_blocks = torch.stack(dK.split(Bc, dim=0), dim=0)
            dV_Tc_blocks = torch.stack(dV.split(Bc, dim=0), dim=0)
            Tc = dK_Tc_blocks.size(0)

            for j in range(Tc):
                Kj, Vj = K_Tc_blocks[j], V_Tc_blocks[j]
                dKj, dVj = torch.zeros_like(Kj), torch.zeros_like(Vj)

                for i in range(Tr):
                    Qi = Q_Tr_blocks[i]
                    Oi = O_Tr_blocks[i]
                    dOi = dO_Tr_blocks[i]
                    dQi = dQ_Tr_blocks[i]
                    Li = L_Tr_blocks[i]
                    Di = D_Tr_blocks[i]

                    Sij = torch.matmul(Qi, Kj.transpose(0,1))
                    
                    # add scale
                    Sij.mul_(scale)

                    Pij = torch.exp(Sij - Li.unsqueeze(-1))
                    dVj.add_(torch.matmul(Pij.transpose(0,1), dOi))
                    dPij = torch.matmul(dOi, Vj.transpose(0,1))
                    dSij = Pij * (dPij - Di.unsqueeze(-1))
                    
                    # add scale
                    dSij.mul_(scale)

                    dQi.add_(torch.matmul(dSij, Kj))
                    dKj.add_(torch.matmul(dSij.transpose(0,1), Qi))

                    dQ_Tr_blocks[i,:,:] = dQi
                dK_Tc_blocks[j,:,:], dV_Tc_blocks[j,:,:] = dKj, dVj
            
            dQuery[b,h,:,:] = dQ_Tr_blocks.reshape(-1, dQ_Tr_blocks.shape[-1])
            dKey[b,h,:,:] = dK_Tc_blocks.reshape(-1, dK_Tc_blocks.shape[-1])
            dValue[b,h,:,:] = dV_Tc_blocks.reshape(-1, dV_Tc_blocks.shape[-1])
    
    return dQuery, dKey, dValue



def attention(query, key, value, scale=None, mask=None, if_causal=True):
    attention_scores = torch.matmul(query, key.transpose(-2, -1))

    scale = scale if scale is not None else key.shape[-1] ** -0.5
    attention_scores = attention_scores * scale
    
    if if_causal:
        causal_mask = torch.tril(torch.ones(attention_scores.shape[-2:], device=attention_scores.device))
        if mask is None:
            attention_scores = attention_scores.masked_fill(causal_mask==0, float('-inf'))

    if mask is not None and not if_causal:
        attention_scores = attention_scores.masked_fill(mask==0, float('-inf'))
    
    attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


def eval_acc():
    # Setup
    torch.manual_seed(0)
    BS, dim, heads, seq_len = 2, 4096, 8, 1024
    Br, Bc = 32, 64
    if_causal = False
    df = torch.float64
    # Test input
    q = torch.randn(BS, heads, seq_len, dim, requires_grad=True).to(df).cuda()
    k = torch.randn(BS, heads, seq_len, dim, requires_grad=True).to(df).cuda()
    v = torch.randn(BS, heads, seq_len, dim, requires_grad=True).to(df).cuda()

    # Forward pass
    out_standard, _ = attention(q, k, v, if_causal=if_causal)
    out_flash, logsumexp = flashattn_forward(q, k, v, Br, Bc, if_causal=if_causal)

    # Calculate absolute differences
    abs_diff = torch.abs(out_standard - out_flash).max().item()
    print(f'Max absolute difference between standard and flash attention forward outputs: {abs_diff:.15f}')

    dOut = torch.randn_like(out_standard)

    # Backward pass
    dQ_standard, dK_standard, dV_standard = torch.autograd.grad(outputs=out_standard, inputs=[q, k, v], grad_outputs=dOut, create_graph=True)
    dQ_flash, dK_flash, dV_flash = flashattn_backward(dOut, q, k, v, out_flash, logsumexp, Br, Bc, if_causal=if_causal)

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


if __name__=="__main__":
    eval_acc()
