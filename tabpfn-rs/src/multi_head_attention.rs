pub(crate) struct MultiHeadAttention<'a> {
    input_size: usize,
    output_size: usize,
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

    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        let (q, k, v) = self.compute_qkv(input);
        let attention = self.compute_attention_heads(&q, &k, &v);
        for i in 0..self.output_size {
            output[i] = self.w_o[i] * attention[i];
        }
    }
}