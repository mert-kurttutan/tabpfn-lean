pub(crate) struct Linear<'a> {
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

    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        for i in 0..self.output_size {
            output[i] = self.weights[i * self.input_size..(i + 1) * self.input_size].iter().zip(input.iter()).map(|(w, x)| w * x).sum();
            if let Some(bias) = &self.bias {
                output[i] += bias[i];
            }
        }
    }
} 


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward() {
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let bias = Some(vec![1.0, 2.0]);
        let linear = Linear::new(&weights, bias.as_ref().map(Vec::as_slice), 2, 2);
        let input = vec![1.0, 2.0];
        let mut output = vec![0.0, 0.0];
        linear.forward(&input, &mut output);
        assert_eq!(output, vec![6.0, 13.0]);
    }
}