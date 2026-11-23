use super::{attention::MultiHeadAttention, feedforward::FeedForward};
use burn::nn::{Dropout, DropoutConfig, LayerNorm, LayerNormConfig};
use burn::tensor::{Bool, Device, Tensor};

#[derive(burn::module::Module, Debug)]
pub struct TransformerBlock {
    attention: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    dropout: Dropout,
}
impl TransformerBlock {
    pub fn new(d_model: usize, heads: usize, ff: usize, dropout: f64, device: &Device) -> Self {
        Self {
            attention: MultiHeadAttention::new(d_model, heads, dropout, device),
            feed_forward: FeedForward::new(d_model, ff, dropout, device),
            norm1: LayerNormConfig::new(d_model).init(device),
            norm2: LayerNormConfig::new(d_model).init(device),
            dropout: DropoutConfig::new(dropout).init(),
        }
    }
    pub fn forward(&self, input: Tensor<3>, mask: Option<Tensor<4, Bool>>) -> Tensor<3> {
        let attended = self.dropout.forward(
            self.attention
                .forward(self.norm1.forward(input.clone()), mask),
        );
        let x = input + attended;
        let fed = self
            .dropout
            .forward(self.feed_forward.forward(self.norm2.forward(x.clone())));
        x + fed
    }
}
