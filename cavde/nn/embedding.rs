use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::{Device, Int, Tensor};

#[derive(burn::module::Module, Debug)]
pub struct TokenEmbedding {
    embedding: Embedding,
    #[module(skip)]
    scale: f64,
}
impl TokenEmbedding {
    pub fn new(vocab_size: usize, d_model: usize, device: &Device) -> Self {
        Self {
            embedding: EmbeddingConfig::new(vocab_size, d_model).init(device),
            scale: (d_model as f64).sqrt(),
        }
    }
    pub fn forward(&self, input: Tensor<2, Int>) -> Tensor<3> {
        self.embedding.forward(input) * self.scale
    }
}
