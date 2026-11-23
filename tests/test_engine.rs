use cavde::cavde::{
    data::{collate::ConversationCollator, dataset::ConversationDataset, reader::Sample},
    tokenizers::SimpleTokenizer,
    vocabulary::Vocabulary,
};
#[test]
fn dataset_batch_has_attention_mask_for_each_input_token() {
    let mut vocab = Vocabulary::new();
    vocab.add_text("hello world");
    vocab.build();
    let tokenizer = SimpleTokenizer::new(vocab);
    let samples = vec![Sample {
        input: "hello".into(),
        output: "world".into(),
    }];
    let dataset = ConversationDataset::from_samples(&samples, &tokenizer);
    let batch = ConversationCollator::default()
        .collate(&dataset.samples)
        .unwrap();
    assert_eq!(batch.input_ids[0].len(), batch.attention_mask[0].len());
}
