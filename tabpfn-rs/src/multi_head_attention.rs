use crate::safetensors::MmapedSafetensors;


pub(crate) struct MultiHeadAttention<'a> {
    input_size: usize,
    pub output_size: usize,
    num_heads: usize,
    d_k: usize,
    w_qkv: Option<&'a [f32]>,
    w_o: &'a [f32],
    kv_cache: Vec<f32>,
    is_kv_cached: bool,
}

impl<'a> MultiHeadAttention<'a> {
    pub fn new(w_qkv: Option<&'a [f32]>, w_o: &'a [f32], input_size: usize, output_size: usize, num_heads: usize) -> Self {
        let d_k = input_size / num_heads;
        Self {
            input_size,
            output_size,
            num_heads,
            d_k,
            w_qkv,
            w_o,
            kv_cache: vec![0.0; num_heads * d_k],
            is_kv_cached: false,
        }
    }

    pub fn load_from_safetensors(mmaped_safetensor: &MmapedSafetensors, prefix: &str) -> Self {
        let w_qkv_name = format!("{}._w_qkv", prefix);
        let w_qkv = mmaped_safetensor.get(&w_qkv_name);
        let w_o_name = format!("{}._w_out", prefix);
        let w_o = mmaped_safetensor.get(&w_o_name);
        let input_size = w_qkv.shape()[1];
        let output_size = w_o.shape()[1];
        let num_heads = w_qkv.shape()[0];
        let d_k = input_size / num_heads;
        let w_qkv_data = unsafe {
            std::slice::from_raw_parts(w_qkv.data().as_ptr() as *const f32, w_qkv.shape().iter().product())
        };
        let w_o_data = unsafe {
            std::slice::from_raw_parts(w_o.data().as_ptr() as *const f32, w_o.shape().iter().product())
        };
        Self {
            input_size,
            output_size,
            num_heads,
            d_k,
            w_qkv: Some(w_qkv_data),
            w_o: w_o_data,
            kv_cache: vec![0.0; num_heads * d_k],
            is_kv_cached: false,
        }
    }
    pub fn compute_qkv(
        &self,
        input: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        if self.is_kv_cached {
            let mut q = vec![0.0; self.num_heads * self.d_k];
            let mut k = vec![0.0; self.num_heads * self.d_k];
            let mut v = vec![0.0; self.num_heads * self.d_k];
            for h in 0..self.num_heads {
                for i in 0..self.d_k {
                    q[h * self.d_k + i] = self.w_qkv.unwrap()[h * self.d_k + i] * input[i];
                    k[h * self.d_k + i] = self.kv_cache[h * self.d_k + i];
                    v[h * self.d_k + i] = self.kv_cache[h * self.d_k + i];
                }
            }
            return (q, k, v);
        }
        let mut q = vec![0.0; self.num_heads * self.d_k];
        let mut k = vec![0.0; self.num_heads * self.d_k];
        let mut v = vec![0.0; self.num_heads * self.d_k];
        for h in 0..self.num_heads {
            for i in 0..self.d_k {
                q[h * self.d_k + i] = self.w_qkv.unwrap()[h * self.d_k + i] * input[i];
            }
        }
        (q, k, v)
    }


    pub fn compute_attention_heads(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
    ) -> Vec<f32> {
        let mut attention = vec![0.0; self.num_heads * self.d_k];
        for h in 0..self.num_heads {
            for i in 0..self.d_k {
                let mut sum = 0.0;
                for j in 0..self.d_k {
                    sum += q[h * self.d_k + j] * k[h * self.d_k + j];
                }
                attention[h * self.d_k + i] = sum / self.d_k as f32;
            }
        }
        attention
    }

    pub fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
        let (q, k, v) = self.compute_qkv(input);
        let attention = self.compute_attention_heads(&q, &k, &v);
        for i in 0..self.output_size {
            output[i] = self.w_o[i] * attention[i];
        }
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
            let weight_name = format!("{}.self_attn_between_features", layer_name);
            // get weight tensor
            MultiHeadAttention::load_from_safetensors(&mmaped_safetensors, &weight_name);


        }
    }
}