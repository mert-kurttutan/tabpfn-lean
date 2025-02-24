use crate::linear::Linear;

pub enum Activation {
    Sigmoid,
    ReLU,
    GeLU,
}
impl Activation {
    pub fn forward(&self, input: &mut[f32]) -> Vec<f32> {
        match self {
            Activation::Sigmoid => input.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect(),
            Activation::ReLU => input.iter().map(|x| x.max(0.0)).collect(),
            Activation::GeLU => input.iter().map(|x| 0.5 * x * (1.0 + (2.0 / std::f32::consts::PI).sqrt() * (0.044715 * x.powi(3)).tanh())).collect(),
        }
    }
}
pub struct MLP<'a> {
    linear1: Linear<'a>,
    activation: Activation,
    linear2: Linear<'a>,
}

impl<'a> MLP<'a> {
    pub fn new(weights1: &'a [f32], bias1: Option<&'a [f32]>, weights2: &'a [f32], bias2: Option<&'a [f32]>, input_size: usize, hidden_size: usize, output_size: usize, activation: Activation) -> Self {
        Self {
            linear1: Linear::new(weights1, bias1, input_size, hidden_size),
            activation,
            linear2: Linear::new(weights2, bias2, hidden_size, output_size),
        }
    }

    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        self.linear1.forward(input, output);
        let mut hidden = vec![0.0; self.linear1.output_size];
        self.activation.forward(&mut hidden);
        self.linear2.forward(&hidden, output);
        self.activation.forward(output);
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_forward() {
        let weights1 = vec![1.0, 2.0, 3.0, 4.0];
        let bias1 = Some(vec![1.0, 2.0]);
        let weights2 = vec![1.0, 2.0, 3.0, 4.0];
        let bias2 = Some(vec![1.0, 2.0]);
        let mlp = MLP::new(&weights1, bias1.as_ref().map(Vec::as_slice), &weights2, bias2.as_ref().map(Vec::as_slice), 2, 2, 2, Activation::ReLU);
        let input = vec![1.0, 2.0];
        let mut output = vec![0.0, 0.0];
        mlp.forward(&input, &mut output);
        assert_eq!(output, vec![1.0, 2.0]);
    }
}