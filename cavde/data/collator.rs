use super::dataset::EncodedSample;

#[derive(Debug, Clone, Default)]
pub struct SimpleCollator {
    pub pad_id: i64,
}

impl SimpleCollator {
    fn pad(&self, sequences: impl IntoIterator<Item = Vec<i64>>) -> Vec<Vec<i64>> {
        let mut sequences: Vec<Vec<i64>> = sequences.into_iter().collect();
        let max_len = sequences.iter().map(Vec::len).max().unwrap_or(0);
        for sequence in &mut sequences {
            sequence.resize(max_len, self.pad_id);
        }
        sequences
    }
    pub fn collate(&self, batch: &[EncodedSample]) -> (Vec<Vec<i64>>, Vec<Vec<i64>>) {
        (
            self.pad(batch.iter().map(|x| x.input_ids.clone())),
            self.pad(batch.iter().map(|x| x.labels.clone())),
        )
    }
}
