from torch import nn
import torch

def test_against_torch():

    # for layernorm

    normalized_count = 133
    normalized_shape = 127
    total_size = normalized_count * normalized_shape
    eps = 1e-5
    x = torch.randn((normalized_count, normalized_shape), dtype=torch.float32)
    x_c = x.detach().clone()
    ln = nn.LayerNorm(normalized_shape, eps=eps)
    y = ln(x)
    y_c = torch.ones_like(y)
    from tabpfn_lean.tabpfn_lean_rs import layer_norm_functional
    layer_norm_functional(
        x_c.data_ptr(), y_c.data_ptr(), 
        # ln.weight.data_ptr(), 
        total_size, normalized_shape)
    # print(f"y_c: {y_c}, size: {y_c.size()}")
    for a,b in zip(y.flatten(), y_c.flatten()):
        rel_error = (a-b).abs() / a.abs()
        abs_error = (a-b).abs()
        if rel_error.min(abs_error) > 1e-5:
            print(f"a: {a}, b: {b}, diff: {a-b}")
            break


if __name__ == "__main__":
    test_against_torch()