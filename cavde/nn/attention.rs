use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::tensor::{Bool, Device, Tensor};

#[derive(burn::module::Module, Debug)]
pub struct MultiHeadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    dropout: Dropout,
    #[module(skip)]
    heads: usize,
    #[module(skip)]
    head_dim: usize,
}
impl MultiHeadAttention {
    pub fn new(d_model: usize, heads: usize, dropout: f64, device: &Device) -> Self {
        assert!(heads > 0, "heads must be greater than zero");
        assert_eq!(
            d_model % heads,
            0,
            "d_model must be divisible by the number of heads"
        );
        Self {
            q_proj: LinearConfig::new(d_model, d_model).init(device),
            k_proj: LinearConfig::new(d_model, d_model).init(device),
            v_proj: LinearConfig::new(d_model, d_model).init(device),
            out_proj: LinearConfig::new(d_model, d_model).init(device),
            dropout: DropoutConfig::new(dropout).init(),
            heads,
            head_dim: d_model / heads,
        }
    }
    // True mask values indicate key positions that must be ignored.
    pub fn forward(&self, input: Tensor<3>, mask: Option<Tensor<4, Bool>>) -> Tensor<3> {
        let [batch, seq, _] = input.dims();
        let q = self
            .q_proj
            .forward(input.clone())
            .reshape([batch, seq, self.heads, self.head_dim])
            .swap_dims(1, 2);
        let k = self
            .k_proj
            .forward(input.clone())
            .reshape([batch, seq, self.heads, self.head_dim])
            .swap_dims(1, 2);
        let v = self
            .v_proj
            .forward(input)
            .reshape([batch, seq, self.heads, self.head_dim])
            .swap_dims(1, 2);
        let mut scores = q.matmul(k.swap_dims(2, 3)) * (1.0_f32 / (self.head_dim as f32).sqrt());
        if let Some(mask) = mask {
            scores = scores.mask_fill(mask, f32::NEG_INFINITY);
        }
        let attention = self
            .dropout
            .forward(burn::tensor::activation::softmax(scores, 3));
        let output =
            attention
                .matmul(v)
                .swap_dims(1, 2)
                .reshape([batch, seq, self.heads * self.head_dim]);
        self.out_proj.forward(output)
    }
}
