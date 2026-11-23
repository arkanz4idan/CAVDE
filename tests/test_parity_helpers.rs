use cavde::cavde::{
    data::preprocess::iter_dataset,
    logger::create_logger,
    metrics::{token_accuracy, token_accuracy_from_logits},
};
use std::fs;

#[test]
fn preprocess_txt_skips_blank_and_malformed_lines_like_python() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sample.txt");
    fs::write(&path, "hello|world\nignored line\n\nfoo|bar|baz\n").unwrap();
    let records = iter_dataset(&path).unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0]["input"], "hello");
    assert_eq!(records[0]["output"], "world");
    assert_eq!(records[1]["output"], "bar|baz");
}

#[test]
fn token_accuracy_accepts_ids_and_raw_logits() {
    assert_eq!(token_accuracy(&[1, 2, 0], &[1, 0, 0]), 2.0 / 3.0);
    let logits = [0.1, 0.9, 0.2, 0.8, 0.7, 0.3];
    assert_eq!(
        token_accuracy_from_logits(&logits, &[1, 0, 0], 2).unwrap(),
        2.0 / 3.0
    );
}

#[test]
fn logger_creates_and_appends_train_log() {
    let dir = tempfile::tempdir().unwrap();
    let logger = create_logger(dir.path()).unwrap();
    logger.info("test message").unwrap();
    let contents = fs::read_to_string(dir.path().join("train.log")).unwrap();
    assert!(contents.contains("INFO : test message"));
    assert_eq!(logger.name(), "CAVDE");
}
