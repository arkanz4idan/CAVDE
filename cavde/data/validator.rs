use super::reader::Sample;
use serde_json::Value;

#[derive(Debug, Clone, Default)]
pub struct DatasetValidator;

impl DatasetValidator {
    /// Validate decoded JSON records before converting them into typed samples.
    pub fn validate_records(&self, records: &[Value]) -> (Vec<Sample>, Vec<String>) {
        let mut valid = Vec::new();
        let mut errors = Vec::new();
        for (index, record) in records.iter().enumerate() {
            let line = index + 1;
            let Some(object) = record.as_object() else {
                errors.push(format!("Line {line}: sample must be a JSON object."));
                continue;
            };
            let Some(input) = object.get("input").and_then(Value::as_str) else {
                errors.push(format!("Line {line}: missing 'input'."));
                continue;
            };
            let Some(output) = object.get("output").and_then(Value::as_str) else {
                errors.push(format!("Line {line}: missing 'output'."));
                continue;
            };
            if input.trim().is_empty() {
                errors.push(format!("Line {line}: empty input."));
                continue;
            }
            if output.trim().is_empty() {
                errors.push(format!("Line {line}: empty output."));
                continue;
            }
            valid.push(Sample {
                input: input.to_owned(),
                output: output.to_owned(),
            });
        }
        (valid, errors)
    }

    /// Validate typed samples; missing fields are impossible at this stage.
    pub fn validate(&self, samples: &[Sample]) -> (Vec<Sample>, Vec<String>) {
        let mut valid = Vec::new();
        let mut errors = Vec::new();
        for (index, sample) in samples.iter().enumerate() {
            if sample.input.trim().is_empty() {
                errors.push(format!("Line {}: empty input.", index + 1));
                continue;
            }
            if sample.output.trim().is_empty() {
                errors.push(format!("Line {}: empty output.", index + 1));
                continue;
            }
            valid.push(sample.clone());
        }
        (valid, errors)
    }
}
