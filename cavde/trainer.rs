use super::{
    checkpoint::Checkpoint, data::dataloader::DataLoader, engine::TrainingEngine,
    scheduler::LearningRateScheduler,
};

#[derive(Debug, Clone)]
pub struct EpochReport {
    pub epoch: usize,
    pub batches: usize,
    pub average_loss: f64,
    pub learning_rate: f64,
}

pub struct Trainer {
    pub epochs: usize,
    pub start_epoch: usize,
    pub best_loss: f64,
    pub validation_enabled: bool,
}

impl Trainer {
    pub fn new(epochs: usize) -> Self {
        Self {
            epochs,
            start_epoch: 0,
            best_loss: f64::INFINITY,
            validation_enabled: false,
        }
    }

    pub fn with_validation(mut self) -> Self {
        self.validation_enabled = true;
        self
    }
    pub fn with_start_epoch(mut self, start_epoch: usize) -> Self {
        self.start_epoch = start_epoch;
        self
    }

    pub fn train(
        &mut self,
        engine: &mut TrainingEngine,
        dataloader: &mut DataLoader,
        scheduler: &mut LearningRateScheduler,
        checkpoint: &Checkpoint,
    ) -> anyhow::Result<Vec<EpochReport>> {
        anyhow::ensure!(
            self.start_epoch <= self.epochs,
            "start epoch exceeds configured epochs"
        );
        anyhow::ensure!(!dataloader.is_empty(), "dataset produced no batches");

        let mut reports = Vec::new();
        for epoch_index in self.start_epoch..self.epochs {
            dataloader.set_epoch(epoch_index as u64);
            let mut total_loss = 0.0_f64;
            let mut batch_count = 0_usize;

            for batch in dataloader.by_ref() {
                let loss = engine.train_batch(&batch)?;
                anyhow::ensure!(
                    loss.is_finite(),
                    "non-finite loss at epoch {}, batch {}",
                    epoch_index + 1,
                    batch_count + 1
                );
                total_loss += loss as f64;
                batch_count += 1;
            }

            anyhow::ensure!(
                batch_count > 0,
                "epoch {} produced no batches",
                epoch_index + 1
            );
            let average_loss = total_loss / batch_count as f64;
            let validation_loss = if self.validation_enabled {
                let validator = super::validator::ValidationEngine::new();
                let loss = validator.validate(engine, dataloader)?;
                log::info!("Epoch {} | validation_loss={:.6}", epoch_index + 1, loss);
                Some(loss)
            } else {
                None
            };
            let checkpoint_loss = validation_loss.unwrap_or(average_loss);

            scheduler.step();
            let learning_rate = scheduler.current_lr();
            engine.set_learning_rate(learning_rate);

            checkpoint.save_training_state(engine, epoch_index + 1, checkpoint_loss, "last")?;
            if checkpoint_loss < self.best_loss {
                self.best_loss = checkpoint_loss;
                checkpoint.save_training_state(engine, epoch_index + 1, checkpoint_loss, "best")?;
            }

            reports.push(EpochReport {
                epoch: epoch_index + 1,
                batches: batch_count,
                average_loss,
                learning_rate,
            });
            log::info!(
                "Epoch {}/{} | batches={} | train_loss={:.6} | lr={:.8}",
                epoch_index + 1,
                self.epochs,
                batch_count,
                average_loss,
                learning_rate
            );
        }
        Ok(reports)
    }
}
