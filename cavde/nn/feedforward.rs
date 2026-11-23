use burn::nn::{Dropout, DropoutConfig, Gelu, Linear, LinearConfig};
use burn::tensor::{Device, Tensor};

#[derive(burn::module::Module, Debug)]
pub struct FeedForward {
    input: Linear,
    activation: Gelu,
    dropout1: Dropout,
    output: Linear,
    dropout2: Dropout,
}
impl FeedForward {
    pub fn new(d_model: usize, hidden_size: usize, dropout: f64, device: &Device) -> Self {
        Self {
            input: LinearConfig::new(d_model, hidden_size).init(device),
            activation: Gelu::new(),
            dropout1: DropoutConfig::new(dropout).init(),
            output: LinearConfig::new(hidden_size, d_model).init(device),
            dropout2: DropoutConfig::new(dropout).init(),
        }
    }
    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let x = self.activation.forward(self.input.forward(input));
        let x = self.dropout1.forward(x);
        self.dropout2.forward(self.output.forward(x))
    }
}
