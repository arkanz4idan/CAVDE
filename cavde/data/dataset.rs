use crate::cavde::{
    data::reader::{DatasetReader, Sample},
    tokenizers::SimpleTokenizer,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedSample {
    pub input_ids: Vec<i64>,
    pub labels: Vec<i64>,
    pub attention_mask: Vec<i64>,
}

#[derive(Debug, Clone)]
pub struct ConversationDataset {
    pub samples: Vec<EncodedSample>,
}

impl ConversationDataset {
    pub fn from_samples(samples: &[Sample], tokenizer: &SimpleTokenizer) -> Self {
        let bos = *tokenizer.vocab.token_to_id.get("<BOS>").unwrap_or(&2);
        let eos = *tokenizer.vocab.token_to_id.get("<EOS>").unwrap_or(&3);
        let sep = *tokenizer.vocab.token_to_id.get("<SEP>").unwrap_or(&4);
        let samples = samples
            .iter()
            .map(|sample| {
                let mut sequence = vec![bos];
                sequence.extend(tokenizer.encode_with_special_tokens(&sample.input, false));
                sequence.push(sep);
                sequence.extend(tokenizer.encode_with_special_tokens(&sample.output, false));
                sequence.push(eos);
                let input_ids = sequence[..sequence.len() - 1].to_vec();
                let labels = sequence[1..].to_vec();
                let attention_mask = vec![1; input_ids.len()];
                EncodedSample {
                    input_ids,
                    labels,
                    attention_mask,
                }
            })
            .collect();
        Self { samples }
    }
    pub fn from_paths(
        paths: &[std::path::PathBuf],
        tokenizer: &SimpleTokenizer,
    ) -> anyhow::Result<Self> {
        let mut samples = Vec::new();
        for path in paths {
            samples.extend(DatasetReader::new(path).read_all()?);
        }
        Ok(Self::from_samples(&samples, tokenizer))
    }
    pub fn len(&self) -> usize {
        self.samples.len()
    }
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}
