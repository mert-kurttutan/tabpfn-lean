use crate::mlp::MLP;
use crate::linear::Linear;
use crate::layer_norm::LayerNorm;
use crate::multi_head_attention::MultiHeadAttention;

pub struct PerFeatureEncoderLayer<'a> {
    self_attn_between_features: MultiHeadAttention<'a>,
    self_attn_between_items: MultiHeadAttention<'a>,
    mlp_1: MLP<'a>,
    mlp_2: MLP<'a>,
    norm1: LayerNorm<'a>,
    norm2: LayerNorm<'a>,
}