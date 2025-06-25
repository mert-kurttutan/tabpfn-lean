use crate::safetensors::MmapedSafetensors;

pub struct Linear<'a> {
    weights: &'a [f32],
    bias: Option<&'a [f32]>,
    input_size: usize,
    pub(crate) output_size: usize,
}

impl<'a> Linear<'a> {
    pub fn new(weights: &'a [f32], bias: Option<&'a [f32]>, input_size: usize, output_size: usize) -> Self {
        Self {
            weights,
            bias,
            input_size,
            output_size,
        }
    }

    pub fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
        for b in 0..batch_size {
            for i in 0..self.output_size {
                let mut cum = 0.0;
                for j in 0..self.input_size {
                    cum += self.weights[i * self.input_size + j] * input[b * self.input_size + j];
                }
                output[b * self.output_size + i] = cum;
                if let Some(bias) = self.bias {
                    output[b * self.output_size + i] += bias[i];
                }
            }
        }
    }

    pub fn load_from_safetensors(mmaped_safetensor: &MmapedSafetensors, prefix: &str) -> Self {
        let w_name = format!("{}.weight", prefix);
        let bias_name = format!("{}.bias", prefix);
        let weights = mmaped_safetensor.get(&w_name);
        let bias = mmaped_safetensor.get(&bias_name);
        let input_size = weights.shape()[0];
        let output_size = weights.shape()[1];

        let w_data = unsafe {
            std::slice::from_raw_parts(weights.data().as_ptr() as *const f32, weights.shape().iter().product())
        };

        let b_data = unsafe {
            std::slice::from_raw_parts(bias.data().as_ptr() as *const f32, bias.shape().iter().product())
        };

        Self {
            weights: w_data,
            bias: Some(b_data),
            input_size,
            output_size,
        }
    }

    pub fn load_from_safetensors_no_bias(mmaped_safetensor: &MmapedSafetensors, prefix: &str) -> Self {
        let w_name = format!("{}.weight", prefix);
        let weights = mmaped_safetensor.get(&w_name);
        let input_size = weights.shape()[0];
        let output_size = weights.shape()[1];

        let w_data = unsafe {
            std::slice::from_raw_parts(weights.data().as_ptr() as *const f32, weights.shape().iter().product())
        };

        Self {
            weights: w_data,
            bias: None,
            input_size,
            output_size,
        }
    }
    // pub fn forward_functional(&self, input: &[f32], output: &mut [f32], weight: &[f32], bias: Option<&[f32]>) {
    //     for i in 0..self.output_size {
    //         for j in 0..self.input_size {
    //             output[i] += weight[i * self.input_size + j] * input[j];
    //         }
    //         if let Some(bias) = bias {
    //             output[i] += bias[i];
    //         }
    //     }
    // }
}


pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut cum = 0.0;
            for l in 0..k {
                cum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = cum;
        }
    }
}

pub fn matmul_t(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut cum = 0.0;
            for l in 0..k {
                cum += a[i * k + l] * b[j * k + l];
            }
            c[i * n + j] = cum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn test_linear_forward() {
    //     let weights = vec![1.0, 2.0, 3.0, 4.0];
    //     let bias = Some(vec![1.0, 2.0]);
    //     let linear = Linear::new(&weights, bias.as_ref().map(Vec::as_slice), 2, 2);
    //     let input = vec![1.0, 2.0];
    //     let mut output = vec![0.0, 0.0];
    //     linear.forward(&input, &mut output);
    //     assert_eq!(output, vec![6.0, 13.0]);
    // }
}