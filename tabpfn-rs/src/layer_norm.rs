pub struct LayerNorm<'a> {
    scale: Option<&'a [f32]>,
    bias: Option<&'a [f32]>,
    normalized_size: usize,
    eps: f32,
}

impl<'a> LayerNorm<'a> {
    pub fn new(scale: Option<&'a [f32]>, bias: Option<&'a [f32]>, normalized_size: usize) -> Self {
        Self {
            scale,
            bias,
            normalized_size,
            eps: 1e-5,
        }
    }

    pub fn forward(&self, input: &[f32], output: &mut [f32], total_size: usize) {
        let normalized_count = total_size / self.normalized_size;
        for i in 0..normalized_count {
            let mean = input[i * self.normalized_size..(i + 1) * self.normalized_size].iter().sum::<f32>() / self.normalized_size as f32;
            let variance = input[i * self.normalized_size..(i + 1) * self.normalized_size].iter().map(|x| (x - mean).powi(2)).sum::<f32>() / self.normalized_size as f32;
            for j in 0..self.normalized_size {
                output[i * self.normalized_size + j] = (input[i * self.normalized_size + j] - mean) / (variance+self.eps).sqrt();// * self.scale[j];
                if let Some(bias) = self.bias {
                    output[i * self.normalized_size + j] += bias[j];
                }
            }
        }
    }
}