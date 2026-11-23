use cavde::cavde::data::{reader::Sample, validator::DatasetValidator};
#[test]
fn rejects_empty_sample_fields() {
    let samples = vec![
        Sample {
            input: " ".into(),
            output: "answer".into(),
        },
        Sample {
            input: "question".into(),
            output: "answer".into(),
        },
    ];
    let (valid, errors) = DatasetValidator.validate(&samples);
    assert_eq!(valid.len(), 1);
    assert_eq!(errors.len(), 1);
}
