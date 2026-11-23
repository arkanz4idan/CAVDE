use super::{config::Config, nn::CAVDEModel, tokenizers::SimpleTokenizer, vocabulary::Vocabulary};
use burn::tensor::Device;

pub fn create_tokenizer_from_vocab(vocab: Vocabulary) -> SimpleTokenizer {
    SimpleTokenizer::new(vocab)
}
pub fn create_tokenizer() -> SimpleTokenizer {
    let mut vocab = Vocabulary::new();
    if let Ok(content) = std::fs::read_to_string("datasets/chat.txt") {
        for line in content.lines() {
            if let Some((input, output)) = line.split_once('|') {
                vocab.add_text(input);
                vocab.add_text(output);
            }
        }
    }
    vocab.build();
    SimpleTokenizer::new(vocab)
}
pub fn create_model(vocab_size: usize, config: &Config, device: &Device) -> CAVDEModel {
    CAVDEModel::new(
        vocab_size,
        config.d_model,
        config.num_layers,
        config.num_heads,
        config.feed_forward,
        config.dropout,
        config.max_sequence_length,
        device,
    )
}
