use burn::nn::loss::CrossEntropyLossConfig;
pub fn create_loss() -> CrossEntropyLossConfig {
    CrossEntropyLossConfig::new()
        .with_pad_tokens(Some(vec![PAD_ID]))
        .with_smoothing(Some(LABEL_SMOOTHING))
}
pub const PAD_ID: usize = 0;
pub const LABEL_SMOOTHING: f32 = 0.1;
