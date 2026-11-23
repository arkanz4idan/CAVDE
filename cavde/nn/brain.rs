use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::{Bool, Device, Int, Tensor};

use super::{base::BaseModel, layers::TransformerBlock, positional::CAVDEPositionalEncoding};

#[derive(Module, Debug)]
pub struct CAVDEModel {
    embedding: Embedding,
    position: CAVDEPositionalEncoding,
    blocks: Vec<TransformerBlock>,
    norm: LayerNorm,
    output: Linear,
}
impl CAVDEModel {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        layers: usize,
        heads: usize,
        feed_forward: usize,
        dropout: f64,
        max_length: usize,
        device: &Device,
    ) -> Self {
        let blocks = (0..layers)
            .map(|_| TransformerBlock::new(d_model, heads, feed_forward, dropout, device))
            .collect();
        Self {
            embedding: EmbeddingConfig::new(vocab_size, d_model).init(device),
            position: CAVDEPositionalEncoding::new(d_model, max_length, device),
            blocks,
            norm: LayerNormConfig::new(d_model).init(device),
            output: LinearConfig::new(d_model, vocab_size).init(device),
        }
    }
    pub fn forward(&self, input: Tensor<2, Int>) -> Tensor<3> {
        self.forward_with_attention_mask(input, None)
    }
    // Optional key-padding mask: 1 means valid token, 0 means padding.
    pub fn forward_with_attention_mask(
        &self,
        input: Tensor<2, Int>,
        attention_mask: Option<Tensor<2, Int>>,
    ) -> Tensor<3> {
        let [batch, seq] = input.dims();
        let mask: Option<Tensor<4, Bool>> =
            attention_mask.map(|values| values.equal_elem(0).reshape([batch, 1, 1, seq]));
        let mut x = self.position.forward(self.embedding.forward(input));
        for block in &self.blocks {
            x = block.forward(x, mask.clone());
        }
        self.output.forward(self.norm.forward(x))
    }
    pub fn parameter_count(&self) -> usize {
        self.num_params()
    }
}
impl BaseModel for CAVDEModel {
    fn forward(&self, input: Tensor<2, Int>) -> Tensor<3> {
        CAVDEModel::forward(self, input)
    }
}
