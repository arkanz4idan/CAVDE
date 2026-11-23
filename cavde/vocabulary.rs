use super::tokenizers::TextNormalizer;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs, path::Path};

pub const SPECIAL_TOKENS: [&str; 5] = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"];

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Vocabulary {
    pub token_to_id: HashMap<String, i64>,
    #[serde(skip)]
    pub id_to_token: HashMap<i64, String>,
    #[serde(skip)]
    counts: HashMap<String, usize>,
    #[serde(skip)]
    token_order: Vec<String>,
    #[serde(skip)]
    normalizer: TextNormalizer,
}

impl Vocabulary {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_text(&mut self, text: &str) {
        for token in self.normalizer.normalize(text).split_whitespace() {
            if !self.counts.contains_key(token) {
                self.token_order.push(token.to_owned());
            }
            *self.counts.entry(token.to_owned()).or_insert(0) += 1;
        }
    }

    /// Mirrors Python Counter.most_common(): descending frequency, stable insertion order for ties.
    pub fn build(&mut self) {
        self.token_to_id.clear();
        self.id_to_token.clear();
        for (index, token) in SPECIAL_TOKENS.iter().enumerate() {
            self.insert((*token).to_owned(), index as i64);
        }
        let mut tokens = self.token_order.clone();
        tokens.sort_by_key(|token| std::cmp::Reverse(*self.counts.get(token).unwrap_or(&0)));
        for token in tokens {
            if !self.token_to_id.contains_key(&token) {
                let index = self.token_to_id.len() as i64;
                self.insert(token, index);
            }
        }
    }

    fn insert(&mut self, token: String, id: i64) {
        self.token_to_id.insert(token.clone(), id);
        self.id_to_token.insert(id, token);
    }

    pub fn encode(&self, text: &str) -> Vec<i64> {
        let unk = *self.token_to_id.get("<UNK>").unwrap_or(&1);
        self.normalizer
            .normalize(text)
            .split_whitespace()
            .map(|token| *self.token_to_id.get(token).unwrap_or(&unk))
            .collect()
    }

    pub fn decode(&self, ids: &[i64]) -> String {
        ids.iter()
            .map(|id| {
                self.id_to_token
                    .get(id)
                    .map(String::as_str)
                    .unwrap_or("<UNK>")
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Writes a JSON object in token-ID order to preserve compatibility with Python vocab files.
    pub fn save(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        if let Some(parent) = path.as_ref().parent() {
            fs::create_dir_all(parent)?;
        }
        let mut entries: Vec<(&String, &i64)> = self.token_to_id.iter().collect();
        entries.sort_by_key(|(_, id)| **id);
        let mut output = String::from("{\n");
        for (index, (token, id)) in entries.iter().enumerate() {
            let key = serde_json::to_string(token)?;
            output.push_str(&format!("  {key}: {id}"));
            if index + 1 != entries.len() {
                output.push(',');
            }
            output.push('\n');
        }
        output.push_str("}\n");
        fs::write(path, output)?;
        Ok(())
    }

    pub fn load(&mut self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        self.token_to_id = serde_json::from_str(&fs::read_to_string(path)?)?;
        self.id_to_token = self
            .token_to_id
            .iter()
            .map(|(token, id)| (*id, token.clone()))
            .collect();
        let mut ordered: Vec<(i64, String)> = self
            .token_to_id
            .iter()
            .filter(|(_, id)| **id >= SPECIAL_TOKENS.len() as i64)
            .map(|(token, id)| (*id, token.clone()))
            .collect();
        ordered.sort_by_key(|(id, _)| *id);
        self.token_order = ordered.into_iter().map(|(_, token)| token).collect();
        self.counts.clear();
        Ok(())
    }

    pub fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }
    pub fn len(&self) -> usize {
        self.vocab_size()
    }
    pub fn is_empty(&self) -> bool {
        self.token_to_id.is_empty()
    }
}
