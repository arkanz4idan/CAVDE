use cavde::cavde::{
    tokenizers::{SimpleTokenizer, base::BaseTokenizer},
    vocabulary::Vocabulary,
};
#[test]
fn decoded_generation_tokens_exclude_control_tokens() {
    let mut vocab = Vocabulary::new();
    vocab.add_text("hello");
    vocab.build();
    let tokenizer = SimpleTokenizer::new(vocab);
    let ids = tokenizer.encode_with_special_tokens("hello", true);
    assert_eq!(tokenizer.decode(&ids), "hello");
}
