

pub fn softmax_in(input: &mut[f32], unit_count: usize, unit_size: usize) {
    for i in 0..unit_count {
        let mut max = input[i * unit_size];
        for j in 1..unit_size {
            if input[i * unit_size + j] > max {
                max = input[i * unit_size + j];
            }
        }
        let mut sum = 0.0;
        for j in 0..unit_size {
            input[i * unit_size + j] = (input[i * unit_size + j] - max).exp();
            sum += input[i * unit_size + j];
        }
        for j in 0..unit_size {
            input[i * unit_size + j] /= sum;
        }
    }
}