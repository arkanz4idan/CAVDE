use super::dataset::EncodedSample;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Batch {
    pub input_ids: Vec<Vec<i64>>,
    pub labels: Vec<Vec<i64>>,
    pub attention_mask: Vec<Vec<i64>>,
}

#[derive(Debug, Clone)]
pub struct ConversationCollator {
    pub pad_id: i64,
}
impl Default for ConversationCollator {
    fn default() -> Self {
        Self { pad_id: 0 }
    }
}

impl ConversationCollator {
    pub fn collate(&self, samples: &[EncodedSample]) -> Option<Batch> {
        if samples.is_empty() {
            return None;
        }
        let max_input = samples.iter().map(|x| x.input_ids.len()).max().unwrap_or(0);
        let max_label = samples.iter().map(|x| x.labels.len()).max().unwrap_or(0);
        let mut input_ids = Vec::with_capacity(samples.len());
        let mut labels = Vec::with_capacity(samples.len());
        let mut attention_mask = Vec::with_capacity(samples.len());
        for sample in samples {
            let mut ids = sample.input_ids.clone();
            let mut target = sample.labels.clone();
            let mut mask = sample.attention_mask.clone();
            ids.resize(max_input, self.pad_id);
            target.resize(max_label, self.pad_id);
            mask.resize(max_input, 0);
            input_ids.push(ids);
            labels.push(target);
            attention_mask.push(mask);
        }
        Some(Batch {
            input_ids,
            labels,
            attention_mask,
        })
    }
}
