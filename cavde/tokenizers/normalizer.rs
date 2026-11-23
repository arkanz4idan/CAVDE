use regex::Regex;
use unicode_normalization::UnicodeNormalization;

#[derive(Debug, Clone, Default)]
pub struct TextNormalizer;

impl TextNormalizer {
    pub fn normalize(&self, text: &str) -> String {
        let normalized: String = text.nfkc().collect();
        let lower = normalized.to_lowercase();
        let punctuation = Regex::new(r"([.,!?;:()])").expect("static regex");
        let spaced = punctuation.replace_all(&lower, " $1 ");
        let whitespace = Regex::new(r"\s+").expect("static regex");
        whitespace.replace_all(&spaced, " ").trim().to_owned()
    }
}
