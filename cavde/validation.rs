use super::{data::dataloader::DataLoader, engine::TrainingEngine, validator::ValidationEngine};
pub struct Validator;
impl Validator {
    pub fn new() -> Self {
        Self
    }
    pub fn validate(
        &self,
        engine: &TrainingEngine,
        dataloader: &mut DataLoader,
    ) -> anyhow::Result<f64> {
        ValidationEngine::new().validate(engine, dataloader)
    }
    pub fn average_loss(losses: &[f64]) -> f64 {
        if losses.is_empty() {
            0.0
        } else {
            losses.iter().sum::<f64>() / losses.len() as f64
        }
    }
}
