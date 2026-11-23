pub trait BaseTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String>;
    fn encode(&self, text: &str) -> Vec<i64>;
    fn decode(&self, ids: &[i64]) -> String;
}
