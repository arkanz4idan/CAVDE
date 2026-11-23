use cavde::cavde::data::reader::{DatasetReader, Sample};
use std::fs;
#[test]
fn reads_pipe_delimited_text() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.txt");
    fs::write(&path, "hello | world\n").unwrap();
    assert_eq!(
        DatasetReader::new(path).read_all().unwrap(),
        vec![Sample {
            input: "hello".into(),
            output: "world".into()
        }]
    );
}
