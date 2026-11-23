use super::{base::BaseTokenizer, normalizer::TextNormalizer};
use crate::cavde::vocabulary::Vocabulary;

#[derive(Debug, Clone)]
pub struct SimpleTokenizer {
    pub vocab: Vocabulary,
    normalizer: TextNormalizer,
}
impl SimpleTokenizer {
    pub fn new(vocab: Vocabulary) -> Self {
        Self {
            vocab,
            normalizer: TextNormalizer,
        }
    }
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        self.normalizer
            .normalize(text)
            .split_whitespace()
            .map(str::to_owned)
            .collect()
    }
    pub fn encode_with_special_tokens(&self, text: &str, add_special_tokens: bool) -> Vec<i64> {
        let mut tokens = self.tokenize(text);
        if add_special_tokens {
            tokens.insert(0, "<BOS>".to_owned());
            tokens.push("<EOS>".to_owned());
        }
        let unk = *self.vocab.token_to_id.get("<UNK>").unwrap_or(&1);
        tokens
            .iter()
            .map(|token| *self.vocab.token_to_id.get(token).unwrap_or(&unk))
            .collect()
    }
    pub fn decode(&self, ids: &[i64]) -> String {
        ids.iter()
            .filter_map(|id| {
                let token = self
                    .vocab
                    .id_to_token
                    .get(id)
                    .map(String::as_str)
                    .unwrap_or("<UNK>");
                if ["<PAD>", "<BOS>", "<EOS>", "<SEP>"].contains(&token) {
                    None
                } else {
                    Some(token)
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}
impl BaseTokenizer for SimpleTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        SimpleTokenizer::tokenize(self, text)
    }
    fn encode(&self, text: &str) -> Vec<i64> {
        self.encode_with_special_tokens(text, false)
    }
    fn decode(&self, ids: &[i64]) -> String {
        SimpleTokenizer::decode(self, ids)
    }
}
