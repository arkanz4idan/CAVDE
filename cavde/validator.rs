use super::{data::dataloader::DataLoader, engine::TrainingEngine};

/// Runs the model in inference-only mode over a data loader.
pub struct ValidationEngine;

impl ValidationEngine {
    pub fn new() -> Self {
        Self
    }

    pub fn validate(
        &self,
        engine: &TrainingEngine,
        dataloader: &mut DataLoader,
    ) -> anyhow::Result<f64> {
        dataloader.reset();
        let mut total_loss = 0.0_f64;
        let mut batches = 0_usize;

        for batch in dataloader {
            let loss = engine.evaluate_batch(&batch)? as f64;
            anyhow::ensure!(loss.is_finite(), "validation produced a non-finite loss");
            total_loss += loss;
            batches += 1;
        }

        anyhow::ensure!(batches > 0, "validation data loader is empty");
        Ok(total_loss / batches as f64)
    }
}
