use super::{
    collate::{Batch, ConversationCollator},
    dataset::ConversationDataset,
};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

#[derive(Debug, Clone)]
pub struct DataLoader {
    dataset: ConversationDataset,
    batch_size: usize,
    shuffle: bool,
    seed: u64,
    epoch: u64,
    batches: Vec<Batch>,
    cursor: usize,
}

impl DataLoader {
    pub fn new(dataset: &ConversationDataset, batch_size: usize, shuffle: bool, seed: u64) -> Self {
        let mut loader = Self {
            dataset: dataset.clone(),
            batch_size: batch_size.max(1),
            shuffle,
            seed,
            epoch: 0,
            batches: Vec::new(),
            cursor: 0,
        };
        loader.rebuild();
        loader
    }

    fn rebuild(&mut self) {
        let mut indices: Vec<usize> = (0..self.dataset.len()).collect();
        if self.shuffle {
            let epoch_seed = self.seed.wrapping_add(self.epoch);
            indices.shuffle(&mut StdRng::seed_from_u64(epoch_seed));
        }
        let collator = ConversationCollator::default();
        self.batches = indices
            .chunks(self.batch_size)
            .filter_map(|chunk| {
                let samples = chunk
                    .iter()
                    .map(|i| self.dataset.samples[*i].clone())
                    .collect::<Vec<_>>();
                collator.collate(&samples)
            })
            .collect();
        self.cursor = 0;
    }

    /// Starts a new epoch with a deterministic but different shuffle order.
    pub fn set_epoch(&mut self, epoch: u64) {
        self.epoch = epoch;
        self.rebuild();
    }

    pub fn len(&self) -> usize {
        self.batches.len()
    }
    pub fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }
    pub fn reset(&mut self) {
        self.cursor = 0;
    }
}

impl Iterator for DataLoader {
    type Item = Batch;
    fn next(&mut self) -> Option<Self::Item> {
        let batch = self.batches.get(self.cursor).cloned();
        if batch.is_some() {
            self.cursor += 1;
        }
        batch
    }
}
