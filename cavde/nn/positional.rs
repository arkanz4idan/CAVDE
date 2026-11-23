use burn::nn::{PositionalEncoding, PositionalEncodingConfig};
use burn::tensor::{Device, Tensor};

#[derive(burn::module::Module, Debug)]
pub struct CAVDEPositionalEncoding {
    encoding: PositionalEncoding,
}
impl CAVDEPositionalEncoding {
    pub fn new(d_model: usize, max_length: usize, device: &Device) -> Self {
        let encoding = PositionalEncodingConfig::new(d_model)
            .with_max_sequence_size(max_length)
            .init(device);
        Self { encoding }
    }
    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        self.encoding.forward(input)
    }
}
