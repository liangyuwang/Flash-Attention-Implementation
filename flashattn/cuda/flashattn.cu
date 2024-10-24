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