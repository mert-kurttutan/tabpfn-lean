use crate::mlp::MLP;
// use crate::linear::Linear;
use crate::layer_norm::LayerNorm;
use crate::multi_head_attention::MultiHeadAttention;

use crate::safetensors::MmapedSafetensors;


pub struct PerFeatureEncoderLayer<'a> {
    self_attn_between_features: MultiHeadAttention<'a>,
    self_attn_between_items: MultiHeadAttention<'a>,
    mlps: [MLP<'a>; 1],
    layer_norms: [LayerNorm<'a>; 4],
}

impl <'a> PerFeatureEncoderLayer<'a> {
    pub fn new(
        self_attn_between_features: MultiHeadAttention<'a>,
        self_attn_between_items: MultiHeadAttention<'a>,
        mlps: [MLP<'a>; 1],
        norm1: LayerNorm<'a>,
        norm2: LayerNorm<'a>,
        norm3: LayerNorm<'a>,
        norm4: LayerNorm<'a>,
    ) -> Self {
        Self {
            self_attn_between_features,
            self_attn_between_items,
            mlps,
            layer_norms: [norm1, norm2, norm3, norm4],
        }
    }

    pub fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
        let mut hidden = vec![0.0; self.self_attn_between_features.output_size * batch_size];
        self.self_attn_between_features.forward(input, &mut hidden, batch_size);
        self.layer_norms[0].forward(&hidden, output, 3);
        self.self_attn_between_items.forward(&output, &mut hidden, batch_size);
        self.layer_norms[1].forward(&hidden, output, 3);
        self.mlps[0].forward(output, &mut hidden, batch_size);
        self.layer_norms[2].forward(&hidden, output, 3);
        // self.mlps[1].forward(&hidden, output, batch_size);
        // self.layer_norms[3].forward(&hidden, output, 3);
    }

    pub fn load_from_safetensors(mmaped_safetensor: &MmapedSafetensors, encoder_layer_prefix: &str) -> Self {
        let self_attn_between_features = MultiHeadAttention::load_from_safetensors(mmaped_safetensor, &format!("{}.self_attn_between_features", encoder_layer_prefix));
        let self_attn_between_items = MultiHeadAttention::load_from_safetensors(mmaped_safetensor, &format!("{}.self_attn_between_items", encoder_layer_prefix));
        let mlp = MLP::load_from_safetensors(mmaped_safetensor, &format!("{}.mlp", encoder_layer_prefix));
        let norm1 = LayerNorm::new(Option::None, Option::None, self_attn_between_features.output_size);
        let norm2 = LayerNorm::new(Option::None, Option::None, self_attn_between_features.output_size);
        let norm3 = LayerNorm::new(Option::None, Option::None, self_attn_between_features.output_size);
        let norm4 = LayerNorm::new(Option::None, Option::None, self_attn_between_features.output_size);
        Self::new(self_attn_between_features, self_attn_between_items, [mlp], norm1, norm2, norm3, norm4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            let encoder_layer = PerFeatureEncoderLayer::load_from_safetensors(&mmaped_safetensors, &layer_name);
            // let input = vec![1.0; 1024];
            // let mut output = vec![0.0; 1024];
            // encoder_layer.forward(&input, &mut output, 1);
        }
    }
}