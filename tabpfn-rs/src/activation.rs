pub enum Activation {
    Sigmoid,
    ReLU,
    GeLU,
}
impl Activation {
    pub fn forward(&self, input: &mut[f32]) {
        match self {
            Activation::Sigmoid => input.iter_mut().for_each(|x| *x = 1.0 / (1.0 + (-*x).exp())),
            Activation::ReLU => input.iter_mut().for_each(|x| *x = x.max(0.0)),
            Activation::GeLU => {
                let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
                for x in input.iter_mut() {
                    let x_val = *x;
                    *x = 0.5 * x_val * (1.0 + (sqrt_2_over_pi*(x_val + 0.044715*x_val*x_val*x_val)).tanh())
                }
            },
        }
    }

    pub fn forward_out(&self, input: &[f32], output: &mut [f32]) {
        match self {
            Activation::Sigmoid => input.iter().zip(output.iter_mut()).for_each(|(x, y)| *y = 1.0 / (1.0 + (-x).exp())),
            Activation::ReLU => input.iter().zip(output.iter_mut()).for_each(|(x, y)| *y = x.max(0.0)),
            Activation::GeLU => input.iter().zip(output.iter_mut()).for_each(
                |(x, y)| *y = 0.5 * x * (1.0 + ((2./std::f32::consts::PI).sqrt()*(x + 0.044715*x*x*x)).tanh()),
            ),
        }
    }
}