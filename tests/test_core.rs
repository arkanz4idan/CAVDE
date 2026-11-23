use std::fs;

use cavde::cavde::{
    data::{
        collate::ConversationCollator,
        dataset::ConversationDataset,
        reader::{DatasetReader, Sample},
    },
    tokenizers::{SimpleTokenizer, base::BaseTokenizer},
    vocabulary::{SPECIAL_TOKENS, Vocabulary},
};

fn tokenizer() -> SimpleTokenizer {
    let mut vocab = Vocabulary::new();
    vocab.add_text("hello world hello");
    vocab.add_text("how are you");
    vocab.build();
    SimpleTokenizer::new(vocab)
}

#[test]
fn vocabulary_reserves_special_token_ids() {
    let mut vocab = Vocabulary::new();
    vocab.add_text("hello world");
    vocab.build();
    for (index, token) in SPECIAL_TOKENS.iter().enumerate() {
        assert_eq!(vocab.token_to_id.get(*token), Some(&(index as i64)));
    }
    assert_eq!(vocab.vocab_size(), 7);
}

#[test]
fn tokenizer_encodes_and_decodes_known_tokens() {
    let tokenizer = tokenizer();
    let ids = tokenizer.encode_with_special_tokens("hello world", true);
    assert_eq!(ids.first(), Some(&2));
    assert_eq!(ids.last(), Some(&3));
    assert_eq!(tokenizer.decode(&ids), "hello world");
}

#[test]
fn tokenizer_maps_unknown_words_to_unk() {
    let tokenizer = tokenizer();
    assert_eq!(tokenizer.encode("not-in-vocabulary"), vec![1]);
}

#[test]
fn dataset_reader_reads_jsonl() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("samples.jsonl");
    fs::write(&path, "{\"input\":\"hello\",\"output\":\"world\"}\n").unwrap();
    let samples = DatasetReader::new(path).read_all().unwrap();
    assert_eq!(
        samples,
        vec![Sample {
            input: "hello".into(),
            output: "world".into()
        }]
    );
}

#[test]
fn dataset_reader_reads_pipe_separated_text() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("samples.txt");
    fs::write(&path, "hello | world\nhow are you | fine\n").unwrap();
    let samples = DatasetReader::new(path).read_all().unwrap();
    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0].input, "hello");
    assert_eq!(samples[0].output, "world");
}

#[test]
fn dataset_reader_reports_malformed_jsonl_line() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("broken.jsonl");
    fs::write(&path, "{not json}\n").unwrap();
    let error = DatasetReader::new(path).read_all().unwrap_err();
    assert!(error.to_string().contains("Line 1"));
}

#[test]
fn conversation_dataset_shifts_targets_by_one_token() {
    let tokenizer = tokenizer();
    let samples = vec![Sample {
        input: "hello".into(),
        output: "world".into(),
    }];
    let dataset = ConversationDataset::from_samples(&samples, &tokenizer);
    let item = &dataset.samples[0];
    assert_eq!(item.input_ids.len(), item.labels.len());
    assert_eq!(item.input_ids[1], tokenizer.vocab.token_to_id["hello"]);
    assert_eq!(item.input_ids[2], 4);
    assert_eq!(item.labels[0], item.input_ids[1]);
    assert_eq!(item.labels.last(), Some(&3));
}

#[test]
fn collator_pads_sequences_and_builds_attention_masks() {
    let tokenizer = tokenizer();
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
    assert_eq!(batch.input_ids.len(), 2);
    assert_eq!(batch.input_ids[0].len(), batch.input_ids[1].len());
    assert_eq!(
        batch.attention_mask[0].iter().filter(|&&x| x == 0).count(),
        1
    );
    assert_eq!(
        batch.attention_mask[1].iter().filter(|&&x| x == 0).count(),
        0
    );
}

#[test]
fn empty_collator_input_returns_none() {
    assert!(ConversationCollator::default().collate(&[]).is_none());
}
