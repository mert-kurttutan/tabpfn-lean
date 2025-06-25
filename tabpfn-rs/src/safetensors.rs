use safetensors::tensor;
use safetensors::tensor as st;
use std::path::Path;
use safetensors::tensor::SafeTensors;
use std::collections::HashMap;


// #[derive(yoke::Yokeable)]
// struct SafeTensors_<'a>(SafeTensors<'a>);

pub struct MmapedSafetensors<'a> {
    pub safetensors: Vec<SafeTensors<'a>>,
    routing: Option<HashMap<String, usize>>,
}

impl<'a> MmapedSafetensors<'a> {
    /// Creates a wrapper around a memory mapped file and deserialize the safetensors header.
    ///
    /// # Safety
    ///
    /// The unsafe is inherited from [`memmap2::MmapOptions`].
    pub unsafe fn new(buffer: &'a[u8]) -> Self {
        // let p = p.as_ref();
        // let file = std::fs::File::open(p).unwrap();
        // let file = memmap2::MmapOptions::new()
        //     .map(&file).unwrap();
        let safetensors = safetensors::SafeTensors::deserialize(buffer).unwrap();
        Self {
            safetensors: vec![safetensors],
            routing: None,
        }
    }

    /// Creates a wrapper around multiple memory mapped file and deserialize the safetensors headers.
    ///
    /// If a tensor name appears in multiple files, the last entry is returned.
    ///
    /// # Safety
    ///
    /// The unsafe is inherited from [`memmap2::MmapOptions`].
    // pub unsafe fn multi<P: AsRef<Path>>(paths: &[P]) -> Self {
    //     let mut routing = HashMap::new();
    //     let mut safetensors = vec![];
    //     for (index, p) in paths.iter().enumerate() {
    //         let p = p.as_ref();
    //         let file = std::fs::File::open(p).unwrap();
    //         let file = memmap2::MmapOptions::new()
    //             .map(&file).unwrap();
    //         let data = safetensors::SafeTensors::deserialize(file.as_ref()).unwrap();
    //         for k in data.get().0.names() {
    //             routing.insert(k.to_string(), index);
    //         }
    //         safetensors.push(data)
    //     }
    //     Self {
    //         safetensors,
    //         routing: Some(routing),
    //     }
    // }

    // pub fn load(&self, name: &str, dev: &Device) -> Tensor {
    //     self.get(name)?.load(dev)
    // }

    pub fn tensors(&self) -> Vec<(String, st::TensorView<'_>)> {
        let mut tensors = vec![];
        for safetensors in self.safetensors.iter() {
            tensors.push(safetensors.tensors())
        }
        tensors.into_iter().flatten().collect()
    }

    pub fn get(&self, name: &str) -> st::TensorView<'_> {
        let index = match &self.routing {
            None => 0,
            Some(routing) => {
                let index = routing.get(name).unwrap();
                *index
            }
        };
        self.safetensors[index].tensor(name).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor() {
        let file_name = "/home/vscode/tabpfn/TabPFN/tabpfn-v2-classifier.sf";
        let file = std::fs::File::open(file_name).unwrap();
        let file = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
        let mmaped_safetensors = unsafe { MmapedSafetensors::new(file.as_ref()) };
        // println!("names: {:?}", mmaped_safetensors.safetensors[0].names());

        for i in 0..12 {
            let layer_prefix = "transformer_encoder.layers.";
            let layer_name = format!("{}{}", layer_prefix, i);
            let weight_name = format!("{}.mlp.linear1.weight", layer_name);
            // get weight tensor
            let weight_tensor = mmaped_safetensors.get(&weight_name);
            let data = weight_tensor.data();
            let shape = weight_tensor.shape();
            // println!("shape: {:?}", shape);
            let elem_count = shape.iter().product();

            use crate::linear::Linear;
            let w_data = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const f32, elem_count)
            };
            let linear = Linear::new(w_data, None, shape[1], shape[0]);


        }

    }
}