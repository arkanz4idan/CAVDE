use burn::{
    module::Module,
    nn::loss::CrossEntropyLoss,
    optim::{AdamWConfig, GradientsParams, ModuleOptimizer},
    tensor::{Device, Int, Tensor, TensorData},
};
use std::path::Path;

use super::{data::collate::Batch, losses::create_loss, nn::CAVDEModel};

/// Executes one optimization step for a CAVDE batch.
pub struct TrainingEngine {
    pub model: CAVDEModel,
    optimizer: ModuleOptimizer,
    criterion: CrossEntropyLoss,
    pub device: Device,
    learning_rate: f64,
}

impl TrainingEngine {
    pub fn new(model: CAVDEModel, device: Device, learning_rate: f64, weight_decay: f32) -> Self {
        let model = model.to_device(&device);
        let optimizer = AdamWConfig::new().with_weight_decay(weight_decay).init();
        let criterion = create_loss().init(&device);
        Self {
            model,
            optimizer,
            criterion,
            device,
            learning_rate,
        }
    }

    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }

    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    pub fn load_optimizer(&mut self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        self.optimizer = self
            .optimizer
            .clone()
            .load(path)
            .map_err(|error| anyhow::anyhow!("{error:?}"))?;
        Ok(())
    }
    pub fn save_optimizer(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        self.optimizer
            .save(path)
            .map_err(|error| anyhow::anyhow!("{error:?}"))
    }

    /// Runs forward, masked cross-entropy, backward, and AdamW update.
    pub fn train_batch(&mut self, batch: &Batch) -> anyhow::Result<f32> {
        let batch_size = batch.input_ids.len();
        anyhow::ensure!(batch_size > 0, "training batch is empty");
        let sequence_len = batch.input_ids[0].len();
        anyhow::ensure!(sequence_len > 0, "training sequence is empty");
        anyhow::ensure!(
            batch.input_ids.iter().all(|row| row.len() == sequence_len),
            "input rows have inconsistent sequence lengths"
        );
        anyhow::ensure!(
            batch.labels.len() == batch_size
                && batch.labels.iter().all(|row| row.len() == sequence_len),
            "label batch shape does not match input batch shape"
        );
        anyhow::ensure!(
            batch.attention_mask.len() == batch_size
                && batch
                    .attention_mask
                    .iter()
                    .all(|row| row.len() == sequence_len),
            "attention mask shape does not match input batch shape"
        );

        let input_values = batch
            .input_ids
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let label_values = batch.labels.iter().flatten().copied().collect::<Vec<_>>();
        let mask_values = batch
            .attention_mask
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let input = Tensor::<2, Int>::from_data(
            TensorData::new(input_values, [batch_size, sequence_len]),
            &self.device,
        );
        let labels = Tensor::<2, Int>::from_data(
            TensorData::new(label_values, [batch_size, sequence_len]),
            &self.device,
        );
        let attention_mask = Tensor::<2, Int>::from_data(
            TensorData::new(mask_values, [batch_size, sequence_len]),
            &self.device,
        );

        let logits = self
            .model
            .forward_with_attention_mask(input, Some(attention_mask));
        let vocab_size = logits.dims()[2];
        let logits = logits.reshape([batch_size * sequence_len, vocab_size]);
        let targets = labels.reshape([batch_size * sequence_len]);
        let loss = self.criterion.forward(logits, targets);

        let loss_data = loss.clone().into_data();
        let loss_value = *loss_data
            .as_slice::<f32>()
            .map_err(|error| anyhow::anyhow!("could not read loss value: {error}"))?
            .first()
            .ok_or_else(|| anyhow::anyhow!("loss tensor is empty"))?;

        let gradients = GradientsParams::from_grads(loss.backward(), &self.model);
        self.model = self
            .optimizer
            .step(self.learning_rate, self.model.clone(), gradients);
        Ok(loss_value)
    }

    /// Computes validation loss without backward propagation or optimizer updates.
    pub fn evaluate_batch(&self, batch: &Batch) -> anyhow::Result<f32> {
        let batch_size = batch.input_ids.len();
        anyhow::ensure!(batch_size > 0, "validation batch is empty");
        let sequence_len = batch.input_ids[0].len();
        anyhow::ensure!(sequence_len > 0, "validation sequence is empty");
        anyhow::ensure!(
            batch.input_ids.iter().all(|row| row.len() == sequence_len),
            "input rows have inconsistent sequence lengths"
        );
        anyhow::ensure!(
            batch.labels.len() == batch_size
                && batch.labels.iter().all(|row| row.len() == sequence_len),
            "label batch shape does not match input batch shape"
        );
        anyhow::ensure!(
            batch.attention_mask.len() == batch_size
                && batch
                    .attention_mask
                    .iter()
                    .all(|row| row.len() == sequence_len),
            "attention mask shape does not match input batch shape"
        );

        let input_values = batch
            .input_ids
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let label_values = batch.labels.iter().flatten().copied().collect::<Vec<_>>();
        let mask_values = batch
            .attention_mask
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let input = Tensor::<2, Int>::from_data(
            TensorData::new(input_values, [batch_size, sequence_len]),
            &self.device,
        );
        let labels = Tensor::<2, Int>::from_data(
            TensorData::new(label_values, [batch_size, sequence_len]),
            &self.device,
        );
        let mask = Tensor::<2, Int>::from_data(
            TensorData::new(mask_values, [batch_size, sequence_len]),
            &self.device,
        );
        let logits = self.model.forward_with_attention_mask(input, Some(mask));
        let vocab_size = logits.dims()[2];
        let loss = self.criterion.forward(
            logits.reshape([batch_size * sequence_len, vocab_size]),
            labels.reshape([batch_size * sequence_len]),
        );
        let data = loss.into_data();
        data.as_slice::<f32>()
            .map_err(|error| anyhow::anyhow!("could not read validation loss: {error}"))?
            .first()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("validation loss tensor is empty"))
    }
}
