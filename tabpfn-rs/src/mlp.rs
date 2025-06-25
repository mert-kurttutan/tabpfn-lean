use crate::linear::Linear;
use crate::Activation;

pub struct MLP<'a> {
    linear1: Linear<'a>,
    activation: Activation,
    linear2: Linear<'a>,
}

use crate::safetensors::MmapedSafetensors;

impl<'a> MLP<'a> {
    pub fn new(weights1: &'a [f32], bias1: Option<&'a [f32]>, weights2: &'a [f32], bias2: Option<&'a [f32]>, input_size: usize, hidden_size: usize, output_size: usize, activation: Activation) -> Self {
        Self {
            linear1: Linear::new(weights1, bias1, input_size, hidden_size),
            activation,
            linear2: Linear::new(weights2, bias2, hidden_size, output_size),
        }
    }

    pub fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
        let mut hidden = vec![0.0; self.linear1.output_size * batch_size];
        self.linear1.forward(input, &mut hidden, batch_size);
        self.activation.forward(&mut hidden);
        self.linear2.forward(&hidden, output, batch_size);
        // self.activation.forward(output);
    }
    pub fn load_from_safetensors(mmaped_safetensor: &MmapedSafetensors, mlp_prefix: &str) -> Self {
        let w1_name = format!("{}.linear1.weight", mlp_prefix);
        let weights1 = mmaped_safetensor.get(&w1_name);
        let bias1 = Option::None;
        let w2_name = format!("{}.linear2.weight", mlp_prefix);
        let weights2 = mmaped_safetensor.get(&w2_name);
        let bias2 = Option::None;
        let input_size = weights1.shape()[0];
        let hidden_size = weights1.shape()[1];
        let output_size = weights2.shape()[1];

        // check alignment of w1 data
        assert_eq!(weights1.data().as_ptr() as usize % 4, 0);

        let w1_data = unsafe {
            std::slice::from_raw_parts(weights1.data().as_ptr() as *const f32, weights1.shape().iter().product())
        };
        let w2_data = unsafe {
            std::slice::from_raw_parts(weights2.data().as_ptr() as *const f32, weights2.shape().iter().product())
        };
        let activation = Activation::ReLU;
        Self {
            linear1: Linear::new(w1_data, bias1, input_size, hidden_size),
            activation,
            linear2: Linear::new(w2_data, bias2, hidden_size, output_size),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn test_mlp_forward() {
    //     let weights1 = vec![1.0, 2.0, 3.0, 4.0];
    //     let bias1 = Some(vec![1.0, 2.0]);
    //     let weights2 = vec![1.0, 2.0, 3.0, 4.0];
    //     let bias2 = Some(vec![1.0, 2.0]);
    //     let mlp = MLP::new(&weights1, bias1.as_ref().map(Vec::as_slice), &weights2, bias2.as_ref().map(Vec::as_slice), 2, 2, 2, Activation::ReLU);
    //     let input = vec![1.0, 2.0];
    //     let mut output = vec![0.0, 0.0];
    //     mlp.forward(&input, &mut output);
    //     assert_eq!(output, vec![1.0, 2.0]);
    // }

    #[test]
    fn test_loading() {
        let file_name = "/home/vscode/tabpfn/TabPFN/tabpfn-v2-classifier.sf";
        let file = std::fs::File::open(file_name).unwrap();
        let file = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
        let mmaped_safetensors = unsafe { MmapedSafetensors::new(file.as_ref()) };
        // println!("names: {:?}", mmaped_safetensors.safetensors[0].names());
        // sort elements in the names vector an print
        let mut names = mmaped_safetensors.safetensors[0].names();
        names.sort();
        // println!("sorted names: {:?}", names);

        for i in 0..12 {
            let layer_prefix = "transformer_encoder.layers.";
            let layer_name = format!("{}{}", layer_prefix, i);
            let weight_name = format!("{}.mlp", layer_name);
            // get weight tensor
            MLP::load_from_safetensors(&mmaped_safetensors, &weight_name);


        }

    }
}