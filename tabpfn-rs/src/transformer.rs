use crate::encoder_layer::PerFeatureEncoderLayer;
use crate::safetensors::MmapedSafetensors;
use crate::mlp::MLP;
use crate::linear::Linear;
pub struct PerFeatureTransformer<'a> {
    encoder_5: Linear<'a>,
    decoder_dict: [Linear<'a>; 2],
    encoder_layers: [PerFeatureEncoderLayer<'a>; 12],
    y_encoder: Linear<'a>,
}

impl <'a> PerFeatureTransformer<'a> {
    pub fn load_from_safetensors(mmaped_safetensor: &MmapedSafetensors, prefix: &str) -> Self {
        // let per_feature_encoder_layers = (0..12).iter(
        //     PerFeatureEncoderLayer::load_from_safetensors(mmaped_safetensor, &format!("{}.layers.{}", prefix, i))
        // )
        let mut array_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

        let encoder_layers = array_index.map(|i| {
            PerFeatureEncoderLayer::load_from_safetensors(mmaped_safetensor, &format!("{}.layers.{}", prefix, i))
        });
        let y_encoder = Linear::load_from_safetensors(mmaped_safetensor, "y_encoder.2.layer");
        let decoder_dict = [
            Linear::load_from_safetensors(mmaped_safetensor, "decoder_dict.standard.0"),
            Linear::load_from_safetensors(mmaped_safetensor, "decoder_dict.standard.2"),
        ];
        let encoder_5 = Linear::load_from_safetensors_no_bias(mmaped_safetensor, "encoder.5.layer");
        PerFeatureTransformer {
            encoder_5,
            decoder_dict,
            encoder_layers,
            y_encoder,

        }

    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_per_feature_transformer_forward() {
        let file_name = "/home/vscode/tabpfn/TabPFN/tabpfn-v2-classifier.sf";
        let file = std::fs::File::open(file_name).unwrap();
        let file = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
        let mmaped_safetensors = unsafe { MmapedSafetensors::new(file.as_ref()) };
        let transformer = PerFeatureTransformer::load_from_safetensors(&mmaped_safetensors, "transformer_encoder");

        // let input = vec![1.0, 2.0, 3.0, 4.0];
        // let mut output = vec![0.0, 0.0, 0.0, 0.0];
        // transformer.forward(&input, &mut output, 1);
        // assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
    }
}