pub(crate) mod linear;
pub(crate) mod mlp;
pub(crate) mod layer_norm;
pub(crate) mod encoder_layer;
pub(crate) mod multi_head_attention;
pub(crate) mod activation;
pub(crate) mod softmax;
pub(crate) mod input_encoder;
pub(crate) mod safetensors;
pub(crate) mod transformer;
pub use mlp::{
    // Activation,
    MLP,
};

use multi_head_attention::MultiHeadAttention;

pub use linear::{
    
    Linear, matmul, matmul_t
};

pub use activation::Activation;

pub use layer_norm::LayerNorm;

pub use softmax::softmax_in;

pub struct Tensor<'a,T> {
    data: &'a mut [T],
    shape: Vec<usize>,
}

impl<'a, T> Tensor<'a, T> {
    pub fn new(data: &'a mut [T], shape: Vec<usize>) -> Self {
        Self {
            data,
            shape,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn data(&mut self) -> &mut [T] {
        self.data
    }
}

pub use input_encoder::sum_out;
pub use input_encoder::mean_out;



pub unsafe fn gelu_fn(x: *mut f32, len: usize) {
    let activation_fn = Activation::GeLU;
    let x_mut_ref = unsafe {
        std::slice::from_raw_parts_mut(x, len)
    };
    activation_fn.forward(x_mut_ref);
}


pub struct PerFeatureEncoderLayer<'a> {
    self_attn_between_features: MultiHeadAttention<'a>,
    self_attn_between_items: MultiHeadAttention<'a>,
    mlp_1: MLP<'a>,
    mlp_2: MLP<'a>,
    norm1: LayerNorm<'a>,
    norm2: LayerNorm<'a>,
}