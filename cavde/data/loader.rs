use super::dataset::EncodedSample;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

pub fn collate_fn(batch: &[EncodedSample]) -> (Vec<Vec<i64>>, Vec<Vec<i64>>) {
    let max_x = batch
        .iter()
        .map(|item| item.input_ids.len())
        .max()
        .unwrap_or(0);
    let max_y = batch
        .iter()
        .map(|item| item.labels.len())
        .max()
        .unwrap_or(0);
    let xs = batch
        .iter()
        .map(|item| {
            let mut values = item.input_ids.clone();
            values.resize(max_x, 0);
            values
        })
        .collect();
    let ys = batch
        .iter()
        .map(|item| {
            let mut values = item.labels.clone();
            values.resize(max_y, 0);
            values
        })
        .collect();
    (xs, ys)
}

pub fn create_dataloader(
    dataset: &[EncodedSample],
    batch_size: usize,
    shuffle: bool,
    seed: u64,
) -> Vec<(Vec<Vec<i64>>, Vec<Vec<i64>>)> {
    let mut indices: Vec<usize> = (0..dataset.len()).collect();
    if shuffle {
        indices.shuffle(&mut StdRng::seed_from_u64(seed));
    }
    indices
        .chunks(batch_size.max(1))
        .map(|chunk| {
            let samples: Vec<EncodedSample> = chunk.iter().map(|i| dataset[*i].clone()).collect();
            collate_fn(&samples)
        })
        .collect()
}
