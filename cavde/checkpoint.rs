use burn::module::Module;
use std::{fs, path::PathBuf};

use super::{engine::TrainingEngine, nn::CAVDEModel};

#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub directory: PathBuf,
}

impl Checkpoint {
    pub fn new(directory: impl Into<PathBuf>) -> anyhow::Result<Self> {
        let directory = directory.into();
        fs::create_dir_all(&directory)?;
        Ok(Self { directory })
    }

    pub fn save_model(&self, model: &CAVDEModel, filename: &str) -> anyhow::Result<()> {
        model
            .clone()
            .save_file(self.directory.join(filename))
            .map_err(|error| anyhow::anyhow!("{error:?}"))
    }

    pub fn load_model(&self, model: CAVDEModel, filename: &str) -> anyhow::Result<CAVDEModel> {
        let path = self.directory.join(filename);
        if !path.exists() {
            anyhow::bail!("Checkpoint not found: {}", path.display());
        }
        model
            .try_load_file(path)
            .map_err(|error| anyhow::anyhow!("{error:?}"))
    }

    /// Saves model parameters, optimizer state, and epoch metadata as separate files.
    pub fn save_training_state(
        &self,
        engine: &TrainingEngine,
        epoch: usize,
        loss: f64,
        name: &str,
    ) -> anyhow::Result<()> {
        fs::create_dir_all(&self.directory)?;
        self.save_model(&engine.model, &format!("{name}.mpk"))?;
        engine.save_optimizer(self.directory.join(format!("{name}.optimizer")))?;
        let metadata = serde_json::json!({
            "epoch": epoch,
            "loss": loss,
            "learning_rate": engine.learning_rate(),
        });
        let temporary = self.directory.join(format!("{name}.json.tmp"));
        let destination = self.directory.join(format!("{name}.json"));
        fs::write(&temporary, serde_json::to_vec_pretty(&metadata)?)?;
        fs::rename(temporary, destination)?;
        Ok(())
    }

    /// Restore the last saved model and optimizer state into an initialized engine.
    pub fn restore_last(&self, engine: &mut TrainingEngine) -> anyhow::Result<serde_json::Value> {
        anyhow::ensure!(self.has_last(), "A complete last checkpoint was not found");
        let metadata = self.last_metadata()?;
        engine.model = self.load_model(engine.model.clone(), "last.mpk")?;
        engine.load_optimizer(self.directory.join("last.optimizer"))?;
        if let Some(learning_rate) = metadata["learning_rate"].as_f64() {
            engine.set_learning_rate(learning_rate);
        }
        Ok(metadata)
    }
    /// Restore model, optimizer, learning rate, and scheduler position from last.
    pub fn restore_training_state(
        &self,
        engine: &mut TrainingEngine,
        scheduler: &mut super::scheduler::LearningRateScheduler,
    ) -> anyhow::Result<usize> {
        anyhow::ensure!(self.has_last(), "complete last checkpoint is not available");
        engine.model = self.load_model(engine.model.clone(), "last.mpk")?;
        engine.load_optimizer(self.directory.join("last.optimizer"))?;
        let metadata = self.last_metadata()?;
        let epoch = metadata["epoch"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("checkpoint epoch is missing or invalid"))?
            as usize;
        let learning_rate = metadata["learning_rate"]
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("checkpoint learning_rate is missing or invalid"))?;
        engine.set_learning_rate(learning_rate);
        scheduler.set_step(epoch);
        Ok(epoch)
    }
    pub fn has_last(&self) -> bool {
        self.directory.join("last.mpk").exists()
            && self.directory.join("last.optimizer").exists()
            && self.directory.join("last.json").exists()
    }

    pub fn metadata(&self, name: &str) -> anyhow::Result<serde_json::Value> {
        let path = self.directory.join(format!("{name}.json"));
        if !path.exists() {
            anyhow::bail!("Checkpoint metadata not found: {}", path.display());
        }
        Ok(serde_json::from_slice(&fs::read(path)?)?)
    }
    pub fn last_metadata(&self) -> anyhow::Result<serde_json::Value> {
        let path = self.directory.join("last.json");
        if !path.exists() {
            anyhow::bail!("Last checkpoint metadata not found");
        }
        Ok(serde_json::from_slice(&fs::read(path)?)?)
    }
}
