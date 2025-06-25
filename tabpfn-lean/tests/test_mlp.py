from torch import nn
import torch

def test_against_torch():
    batch_dim = 1400
    input_dim = 147
    output_dim = 192
    linear = nn.Linear(input_dim, output_dim, bias=False)
    # x = torch.ones((batch_dim, input_dim), dtype=torch.float32)
    # tensor of length batch_dim * input_dim ranging from 0 to batch_dim * input_dim
    # x = torch.arange(0, batch_dim * input_dim, dtype=torch.float32).reshape(batch_dim, input_dim)
    x = torch.randn((batch_dim, input_dim), dtype=torch.float32)
    # linear.weight.data.fill_(1)
    w_c = torch.randn((output_dim, input_dim), dtype=torch.float32)
    linear.weight.data = w_c
    x_c = x.detach().clone()
    y = linear(x)

    y_c = torch.zeros_like(y)

    from tabpfn_lean.tabpfn_lean_rs import linear_functional
    linear_functional(x_c.data_ptr(), y_c.data_ptr(), linear.weight.data_ptr(), batch_dim, input_dim, output_dim)

    # iterate through every lele of of y and y flattened and print if not equal
    for a,b in zip(y.flatten(), y_c.flatten()):
        rel_error = (a-b).abs() / a.abs()
        abs_error = (a-b).abs()
        if rel_error.min(abs_error) > 1e-5:
            print(f"a: {a}, b: {b}, diff: {a-b}")

    # assert torch.allclose(y, y_c, atol=1e-7), f"y: {y}, y_c: {y_c}"



if __name__ == "__main__":
    test_against_torch()