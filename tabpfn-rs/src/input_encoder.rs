use crate::Tensor;
pub fn sum_out(input: &mut Tensor<f32>, output: &mut Tensor<f32>, axis: usize) {
    let sum_stride: usize = input.shape()[axis..].iter().product::<usize>() / input.shape()[axis];
    let sum_size = input.shape()[axis];
    let sum_count: usize = input.shape().iter().product::<usize>() / (sum_size * sum_stride);

    for i in 0..sum_count {
        
        for j in 0..sum_stride {
            let mut sum = 0.0;
            for k in 0..sum_size {
                sum += input.data()[(i * sum_size * sum_stride) + (k * sum_stride) + j];
            }
            output.data()[(i * sum_stride) + j] = sum;
        }
    }
}

pub fn mean_out(input: &mut Tensor<f32>, output: &mut Tensor<f32>, axis: usize) {
    let sum_stride: usize = input.shape()[axis..].iter().product::<usize>() / input.shape()[axis];
    let sum_size = input.shape()[axis];
    let sum_count: usize = input.shape().iter().product::<usize>() / (sum_size * sum_stride);

    for i in 0..sum_count {
        
        for j in 0..sum_stride {
            let mut sum = 0.0;
            for k in 0..sum_size {
                sum += input.data()[(i * sum_size * sum_stride) + (k * sum_stride) + j];
            }
            output.data()[(i * sum_stride) + j] = sum / sum_size as f32;
        }
    }
}