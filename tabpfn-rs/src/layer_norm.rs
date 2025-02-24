pub(crate) struct LayerNorm<'a> {
    scale: &'a [f32],
    bias: &'a [f32],
    input_size: usize,
}

impl<'a> LayerNorm<'a> {
    pub fn new(scale: &'a [f32], bias: &'a [f32], input_size: usize) -> Self {
        Self {
            scale,
            bias,
            input_size,
        }
    }

    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        let mean = input.iter().sum::<f32>() / self.input_size as f32;
        let variance = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / self.input_size as f32;
        for i in 0..self.input_size {
            output[i] = (input[i] - mean) / variance.sqrt() * self.scale[i] + self.bias[i];
        }
    }
}