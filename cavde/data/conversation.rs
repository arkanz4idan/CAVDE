use super::reader::Sample;
use crate::cavde::tokenizers::SimpleTokenizer;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversationItem {
    pub input_ids: Vec<i64>,
    pub labels: Vec<i64>,
}

#[derive(Debug, Clone)]
pub struct ConversationDatasetView {
    pub samples: Vec<ConversationItem>,
}

impl ConversationDatasetView {
    pub fn new(samples: &[Sample], tokenizer: &SimpleTokenizer) -> Self {
        let samples = samples
            .iter()
            .map(|sample| {
                let conversation = format!("{} {}", sample.input, sample.output);
                let tokens = tokenizer.encode_with_special_tokens(&conversation, true);
                ConversationItem {
                    input_ids: tokens[..tokens.len().saturating_sub(1)].to_vec(),
                    labels: tokens.get(1..).unwrap_or_default().to_vec(),
                }
            })
            .collect();
        Self { samples }
    }
    pub fn len(&self) -> usize {
        self.samples.len()
    }
}
