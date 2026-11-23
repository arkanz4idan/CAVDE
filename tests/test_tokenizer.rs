use cavde::cavde::{
    tokenizers::{SimpleTokenizer, base::BaseTokenizer},
    vocabulary::Vocabulary,
};
#[test]
fn tokenizer_round_trips_known_words() {
    let mut vocab = Vocabulary::new();
    vocab.add_text("hello world");
    vocab.build();
    let tokenizer = SimpleTokenizer::new(vocab);
    assert_eq!(
        tokenizer.decode(&tokenizer.encode("hello world")),
        "hello world"
    );
}
