#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>

namespace py = pybind11;

// Forward declaration with default arguments
std::tuple<torch::Tensor, torch::Tensor> flashattn_forward(
    torch::Tensor Query, torch::Tensor Key, torch::Tensor Value,
    int Br, int Bc,
    bool if_causal = true);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> flashattn_backward(
    torch::Tensor dOut,
    torch::Tensor Query, torch::Tensor Key, torch::Tensor Value,
    torch::Tensor Out, torch::Tensor Logsumexp,
    int Br, int Bc,
    bool if_causal = true);


PYBIND11_MODULE(flashattn, m) {
    m.def("forward", &flashattn_forward, "A function that implements the Flash Attention forward pass",
          py::arg("Query"), py::arg("Key"), py::arg("Value"),
          py::arg("Br"), py::arg("Bc"),
          py::arg("if_causal") = true);

    m.def("backward", &flashattn_backward, "A function that implements the Flash Attention backward pass",
          py::arg("dOut"), py::arg("Query"), py::arg("Key"), py::arg("Value"),
          py::arg("Out"), py::arg("Logsumexp"),
          py::arg("Br"), py::arg("Bc"),
          py::arg("if_causal") = true);
}


std::tuple<torch::Tensor, torch::Tensor> flashattn_forward(
    torch::Tensor Query, torch::Tensor Key, torch::Tensor Value, 
    int Br, int Bc, 
    bool if_causal) {
    
    auto BS = Query.size(0);
    auto n_qhead = Query.size(1);
    auto N = Query.size(2);
    auto dim = Query.size(3);
    auto Out = torch::empty_like(Query);
    auto Logsumexp = torch::empty({BS, n_qhead, N}, Query.options());

    // add scale
    c10::optional<float> scale = 1 / std::sqrt(dim);

    // add mask
    torch::Tensor causal_mask;
    if (if_causal) {
        causal_mask = torch::tril(torch::ones({N, N}, Query.options()));
    }

    for (int b = 0; b < BS; ++b) {
        for (int h = 0; h < n_qhead; ++h) {
            auto Q = Query[b][h];
            auto K = Key[b][h];
            auto V = Value[b][h];
            auto O = torch::empty_like(Q);
            auto L = torch::empty({Q.size(0)}, Q.options());

            auto Q_Tr_blocks = torch::stack(Q.split(Br, 0), 0);
            auto O_Tr_blocks = torch::stack(O.split(Br, 0), 0);
            auto L_Tr_blocks = torch::stack(L.split(Br), 0);
            auto Tr = Q_Tr_blocks.size(0);

            auto K_Tc_blocks = torch::stack(K.split(Bc, 0), 0);
            auto V_Tc_blocks = torch::stack(V.split(Bc, 0), 0);
            auto Tc = K_Tc_blocks.size(0);

            for (int i = 0; i < Tr; ++i) {
                auto Qi = Q_Tr_blocks[i];
                auto Oi0 = O_Tr_blocks[i].fill_(0);
                auto li0 = torch::zeros({Br}, Q.options());
                auto mi0 = torch::full_like(li0, -std::numeric_limits<float>::infinity());
                
                torch::Tensor mij_1 = mi0, lij_1 = li0, Oij_1 = Oi0;  // Initialize before loop
                torch::Tensor mij, lij, Oij;  // Declare here to retain values between iterations

                for (int j = 0; j < Tc; ++j) {
                    auto Kj = K_Tc_blocks[j];
                    auto Vj = V_Tc_blocks[j];
                    auto Sij = torch::matmul(Qi, Kj.transpose(0, 1));

                    // add scale
                    Sij.mul_(*scale);

                    // add mask
                    if (if_causal) {
                        auto causal_mask_ij = causal_mask.slice(0, i*Br, (i+1)*Br).slice(1, j*Bc, (j+1)*Bc);
                        Sij.masked_fill_(causal_mask_ij == 0, -std::numeric_limits<float>::infinity());
                    }

                    auto [Sij_rowmax, _] = torch::max(Sij, -1);
                    mij = torch::max(mij_1, Sij_rowmax);
                    auto Pij = torch::exp(Sij - mij.unsqueeze(-1));
                    lij = torch::exp(mij_1 - mij) * lij_1 + torch::sum(Pij, -1);
                    Oij = torch::matmul(torch::diag(torch::exp(mij_1 - mij)), Oij_1) + torch::matmul(Pij, Vj);

                    mij_1 = mij;
                    lij_1 = lij;
                    Oij_1 = Oij;
                }
                auto li_Tc = lij;
                auto Oi_Tc = Oij;
                auto mi_Tc = mij;
                li_Tc = lij;

                auto Oi = torch::matmul(torch::inverse(torch::diag(li_Tc)), Oi_Tc);
                auto Li = mi_Tc + torch::log(li_Tc);

                O_Tr_blocks[i] = Oi.view_as(O_Tr_blocks[i]);
                L_Tr_blocks[i] = Li;
            }
            Out[b][h] = O_Tr_blocks.view_as(Out[b][h]);
            Logsumexp[b][h] = L_Tr_blocks.view_as(Logsumexp[b][h]);
        }
    }
    
    return std::make_tuple(Out, Logsumexp);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> flashattn_backward(
    torch::Tensor dOut,
    torch::Tensor Query, torch::Tensor Key, torch::Tensor Value,
    torch::Tensor Out, torch::Tensor Logsumexp,
    int Br, int Bc,
    bool if_causal) {
    
    auto BS = Query.size(0);
    auto n_qhead = Query.size(1);
    auto N = Query.size(2);
    auto dim = Query.size(3);
    auto dQuery = torch::empty_like(Query);
    auto dKey = torch::empty_like(Key);
    auto dValue = torch::empty_like(Value);

    // Add scale
    c10::optional<float> scale = 1 / std::sqrt(dim);

    // Add mask
    torch::Tensor causal_mask;
    if (if_causal) {
        causal_mask = torch::tril(torch::ones({N, N}, Query.options()));
    }

    for (int b = 0; b < BS; ++b) {
        for (int h = 0; h < n_qhead; ++h) {
            auto Q = Query[b][h];
            auto K = Key[b][h];
            auto V = Value[b][h];
            auto dQ = dQuery[b][h];
            auto dK = dKey[b][h];
            auto dV = dValue[b][h];
            auto O = Out[b][h];
            auto L = Logsumexp[b][h];
            auto dO = dOut[b][h];
            auto D = torch::sum(dO * O, -1);

            auto Q_Tr_blocks = torch::stack(Q.split(Br, 0), 0);
            auto dQ_Tr_blocks = torch::stack(dQ.split(Br, 0), 0);
            auto dO_Tr_blocks = torch::stack(dO.split(Br, 0), 0);
            auto L_Tr_blocks = torch::stack(L.split(Br), 0);
            auto D_Tr_blocks = torch::stack(D.split(Br), 0);
            auto Tr = dQ_Tr_blocks.size(0);

            auto K_Tc_blocks = torch::stack(K.split(Bc, 0), 0);
            auto V_Tc_blocks = torch::stack(V.split(Bc, 0), 0);
            auto dK_Tc_blocks = torch::stack(dK.split(Bc, 0), 0);
            auto dV_Tc_blocks = torch::stack(dV.split(Bc, 0), 0);
            auto Tc = dK_Tc_blocks.size(0);

            for (int j = 0; j < Tc; ++j) {
                auto Kj = K_Tc_blocks[j];
                auto Vj = V_Tc_blocks[j];
                auto dKj = torch::zeros_like(Kj);
                auto dVj = torch::zeros_like(Vj);

                for (int i = 0; i < Tr; ++i) {
                    auto Qi = Q_Tr_blocks[i];
                    auto dOi = dO_Tr_blocks[i];
                    auto dQi = dQ_Tr_blocks[i];
                    auto Li = L_Tr_blocks[i];
                    auto Di = D_Tr_blocks[i];

                    auto Sij = torch::matmul(Qi, Kj.transpose(0, 1));
                    
                    // Add scale
                    Sij.mul_(*scale);

                    // Add mask
                    if (if_causal) {
                        auto causal_mask_ij = causal_mask.slice(0, i * Br, (i + 1) * Br).slice(1, j * Bc, (j + 1) * Bc);
                        Sij.masked_fill_(causal_mask_ij == 0, -std::numeric_limits<float>::infinity());
                    }

                    auto Pij = torch::exp(Sij - Li.unsqueeze(-1));
                    dVj += torch::matmul(Pij.transpose(0, 1), dOi);
                    auto dPij = torch::matmul(dOi, Vj.transpose(0, 1));
                    auto dSij = Pij * (dPij - Di.unsqueeze(-1));
                    
                    // Add scale
                    dSij.mul_(*scale);

                    dQi += torch::matmul(dSij, Kj);
                    dKj += torch::matmul(dSij.transpose(0, 1), Qi);

                    dQ_Tr_blocks[i] = dQi;
                }
                dK_Tc_blocks[j] = dKj;
                dV_Tc_blocks[j] = dVj;
            }
            
            dQuery[b][h] = dQ_Tr_blocks.view_as(dQuery[b][h]);
            dKey[b][h] = dK_Tc_blocks.view_as(dKey[b][h]);
            dValue[b][h] = dV_Tc_blocks.view_as(dValue[b][h]);
        }
    }
    
    return std::make_tuple(dQuery, dKey, dValue);
}

