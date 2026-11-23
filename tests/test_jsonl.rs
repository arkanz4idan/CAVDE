use cavde::cavde::data::reader::{DatasetReader, Sample};
use std::fs;
#[test]
fn reads_jsonl_records() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.jsonl");
    fs::write(&path, "{\"input\":\"hello\",\"output\":\"world\"}\n").unwrap();
    assert_eq!(
        DatasetReader::new(path).read_all().unwrap(),
        vec![Sample {
            input: "hello".into(),
            output: "world".into()
        }]
    );
}
