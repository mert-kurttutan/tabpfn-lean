use tabpfn_rs::{Activation, MLP, Linear};

use pyo3::prelude::*;


#[pyfunction]
unsafe fn gelu_fn(x: usize, len: usize) {
    let activation_fn = Activation::GeLU;
    let x_mut_ptr = x as *mut f32;
    // let len = len as usize;
    let x_mut_ref = unsafe {
        std::slice::from_raw_parts_mut(x_mut_ptr, len)
    };
    activation_fn.forward(x_mut_ref);
}

#[pyfunction]
unsafe fn gelu_fn_out(x: usize, y: usize, len: usize) {
    let activation_fn = Activation::GeLU;
    let x_mut_ptr = x as *mut f32;
    let y_mut_ptr = y as *mut f32;
    let x_mut_ref = unsafe {
        std::slice::from_raw_parts_mut(x_mut_ptr, len)
    };
    let y_mut_ref = unsafe {
        std::slice::from_raw_parts_mut(y_mut_ptr, len)
    };
    activation_fn.forward_out(x_mut_ref, y_mut_ref);
}

#[pyfunction]
unsafe fn mlp_functional(x: usize, y: usize, w1: usize, w2: usize, batch_size: usize, input_size: usize, hidden_size: usize, output: usize, activation: usize) {
    let activation = match activation {
        0 => Activation::GeLU,
        _ => Activation::ReLU,
    };
    let weight1_ref = unsafe {
        std::slice::from_raw_parts_mut(w1 as *mut f32, input_size * hidden_size)
    };
    let weight2_ref = unsafe {
        std::slice::from_raw_parts_mut(w2 as *mut f32, hidden_size * output)
    };
    let mlp = MLP::new(weight1_ref, Option::None, weight2_ref, Option::None, input_size, hidden_size, output, activation);
    let x_mut_ref = unsafe {
        std::slice::from_raw_parts_mut(x as *mut f32, input_size * batch_size)
    };
    let y_mut_ref = unsafe {
        std::slice::from_raw_parts_mut(y as *mut f32, output * batch_size)
    };

    mlp.forward(x_mut_ref, y_mut_ref, batch_size);
}

#[pyfunction]
unsafe fn linear_functional(x: usize, y: usize, w: usize, batch_size: usize, input_size: usize, output_size: usize) {
    let weight_ref = unsafe {
        std::slice::from_raw_parts_mut(w as *mut f32, input_size * output_size)
    };
    let linear = Linear::new(weight_ref, Option::None, input_size, output_size);
    let x_mut_ref = unsafe {
        std::slice::from_raw_parts_mut(x as *mut f32, input_size * batch_size)
    };
    let y_mut_ref = unsafe {
        std::slice::from_raw_parts_mut(y as *mut f32, output_size * batch_size)
    };

    linear.forward(x_mut_ref, y_mut_ref, batch_size);
}

#[pyfunction]
unsafe fn layer_norm_functional(
    x: usize,
    y: usize,
    // w: usize,
    // b: usize,
    total_size: usize,
    normalized_size: usize,
) {
    use tabpfn_rs::LayerNorm;
    // let weight_ref = unsafe {
    //     std::slice::from_raw_parts_mut(w as *mut f32, normalized_size)
    // };
    // let bias_ref = unsafe {
    //     std::slice::from_raw_parts_mut(b as *mut f32, normalized_size)
    // };

    let layer_norm = LayerNorm::new(Option::None, Option::None, normalized_size);
    let x_mut_ref = unsafe {
        std::slice::from_raw_parts_mut(x as *mut f32, total_size)
    };
    let y_mut_ref = unsafe {
        std::slice::from_raw_parts_mut(y as *mut f32, total_size)
    };

    layer_norm.forward(x_mut_ref, y_mut_ref, total_size);
}

#[pyfunction]
unsafe fn matmul(a: usize, b: usize, c: usize, m: usize, n: usize, k: usize) {
    let a_ref = unsafe {
        std::slice::from_raw_parts_mut(a as *mut f32, m * k)
    };
    let b_ref = unsafe {
        std::slice::from_raw_parts_mut(b as *mut f32, k * n)
    };
    let c_ref = unsafe {
        std::slice::from_raw_parts_mut(c as *mut f32, m * n)
    };
    tabpfn_rs::matmul(a_ref, b_ref, c_ref, m, n, k);
}

#[pyfunction]
unsafe fn matmul_t(a: usize, b: usize, c: usize, m: usize, n: usize, k: usize) {
    let a_ref = unsafe {
        std::slice::from_raw_parts_mut(a as *mut f32, m * k)
    };
    let b_ref = unsafe {
        std::slice::from_raw_parts_mut(b as *mut f32, n * k)
    };
    let c_ref = unsafe {
        std::slice::from_raw_parts_mut(c as *mut f32, m * n)
    };
    tabpfn_rs::matmul_t(a_ref, b_ref, c_ref, m, n, k);
}

#[pyfunction]
unsafe fn batch_matmul(a: usize, b: usize, c: usize, batch_size: usize, m: usize, n: usize, k: usize) {
    let a_ref = unsafe {
        std::slice::from_raw_parts_mut(a as *mut f32, batch_size * m * k)
    };
    let b_ref = unsafe {
        std::slice::from_raw_parts_mut(b as *mut f32, batch_size * k * n)
    };
    let c_ref = unsafe {
        std::slice::from_raw_parts_mut(c as *mut f32, batch_size * m * n)
    };
    for b in 0..batch_size {
        tabpfn_rs::matmul(&a_ref[b * m * k..(b + 1) * m * k], &b_ref[b * k * n..(b + 1) * k * n], &mut c_ref[b * m * n..(b + 1) * m * n], m, n, k);
    }
}

#[pyfunction]
unsafe fn batch_matmul_t(a: usize, b: usize, c: usize, batch_size: usize, m: usize, n: usize, k: usize) {
    let a_ref = unsafe {
        std::slice::from_raw_parts_mut(a as *mut f32, batch_size * m * k)
    };
    let b_ref = unsafe {
        std::slice::from_raw_parts_mut(b as *mut f32, batch_size * n * k)
    };
    let c_ref = unsafe {
        std::slice::from_raw_parts_mut(c as *mut f32, batch_size * m * n)
    };
    for b in 0..batch_size {
        tabpfn_rs::matmul_t(&a_ref[b * m * k..(b + 1) * m * k], &b_ref[b * n * k..(b + 1) * n * k], &mut c_ref[b * m * n..(b + 1) * m * n], m, n, k);
    }
}

#[pyfunction]
unsafe fn softmax_in(input: usize, unit_count: usize, unit_size: usize) {
    let input_ref = unsafe {
        std::slice::from_raw_parts_mut(input as *mut f32, unit_count * unit_size)
    };
    tabpfn_rs::softmax_in(input_ref, unit_count, unit_size);
}

use tabpfn_rs::sum_out;

#[pyfunction]
unsafe fn sum(input: usize, output: usize, shape: Vec<usize>, axis: usize) {
    let input_ref = unsafe {
        std::slice::from_raw_parts_mut(input as *mut f32, shape.iter().product())
    };
    let output_ref = unsafe {
        std::slice::from_raw_parts_mut(output as *mut f32, shape.iter().product())
    };
    let mut input_tensor = tabpfn_rs::Tensor::new(input_ref, shape.clone());
    let mut output_tensor = tabpfn_rs::Tensor::new(output_ref, shape.clone());
    sum_out(&mut input_tensor, &mut output_tensor, axis);
}

#[pyfunction]
unsafe fn mean(input: usize, output: usize, shape: Vec<usize>, axis: usize) {
    let input_ref = unsafe {
        std::slice::from_raw_parts_mut(input as *mut f32, shape.iter().product())
    };
    let output_ref = unsafe {
        std::slice::from_raw_parts_mut(output as *mut f32, shape.iter().product())
    };
    let mut input_tensor = tabpfn_rs::Tensor::new(input_ref, shape.clone());
    let mut output_tensor = tabpfn_rs::Tensor::new(output_ref, shape.clone());
    tabpfn_rs::mean_out(&mut input_tensor, &mut output_tensor, axis);
}

#[pymodule]
fn tabpfn_lean_rs(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(gelu_fn, module)?)?;
    module.add_function(wrap_pyfunction!(gelu_fn_out, module)?)?;
    module.add_function(wrap_pyfunction!(mlp_functional, module)?)?;
    module.add_function(wrap_pyfunction!(linear_functional, module)?)?;
    module.add_function(wrap_pyfunction!(layer_norm_functional, module)?)?;
    module.add_function(wrap_pyfunction!(matmul, module)?)?;
    module.add_function(wrap_pyfunction!(matmul_t, module)?)?;
    module.add_function(wrap_pyfunction!(batch_matmul, module)?)?;
    module.add_function(wrap_pyfunction!(batch_matmul_t, module)?)?;
    module.add_function(wrap_pyfunction!(softmax_in, module)?)?;
    module.add_function(wrap_pyfunction!(sum, module)?)?;
    module.add_function(wrap_pyfunction!(mean, module)?)?;
    Ok(())
}
