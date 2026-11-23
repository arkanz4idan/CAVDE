use burn::tensor::{Int, Tensor};

/// Common CAVDE model contract. Burn 0.22 keeps the execution device on tensors,
/// so the model type itself is independent of the selected backend.
pub trait BaseModel {
    fn forward(&self, input: Tensor<2, Int>) -> Tensor<3>;
}
