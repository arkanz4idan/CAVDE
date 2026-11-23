use cavde::cavde::{
    data::{collate::ConversationCollator, dataset::ConversationDataset, reader::Sample},
    tokenizers::SimpleTokenizer,
    vocabulary::Vocabulary,
};
#[test]
fn collated_batch_has_matching_input_label_shapes() {
    let mut vocab = Vocabulary::new();
    vocab.add_text("hello world");
    vocab.build();
    let tokenizer = SimpleTokenizer::new(vocab);
    let samples = vec![
        Sample {
            input: "hello".into(),
            output: "world".into(),
        },
        Sample {
            input: "hello world".into(),
            output: "hello".into(),
        },
    ];
    let dataset = ConversationDataset::from_samples(&samples, &tokenizer);
    let batch = ConversationCollator::default()
        .collate(&dataset.samples)
        .unwrap();
    assert_eq!(batch.input_ids.len(), batch.labels.len());
    assert_eq!(batch.input_ids[0].len(), batch.labels[0].len());
}
