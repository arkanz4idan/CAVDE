use serde_json::Value;
use std::{fs, path::PathBuf};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sample {
    pub input: String,
    pub output: String,
}

#[derive(Debug, Error)]
pub enum DatasetReadError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Unsupported dataset format: {0}")]
    UnsupportedFormat(String),
    #[error("Line {line}: {message}")]
    Line { line: usize, message: String },
    #[error("Invalid JSON dataset: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone)]
pub struct DatasetReader {
    pub path: PathBuf,
}

impl DatasetReader {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
    pub fn read_all(&self) -> Result<Vec<Sample>, DatasetReadError> {
        let suffix = self
            .path
            .extension()
            .and_then(|x| x.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        match suffix.as_str() {
            "txt" => self.read_txt(),
            "jsonl" => self.read_jsonl(),
            "json" => self.read_json(),
            _ => Err(DatasetReadError::UnsupportedFormat(suffix)),
        }
    }
    fn read_txt(&self) -> Result<Vec<Sample>, DatasetReadError> {
        let content = fs::read_to_string(&self.path)?;
        let mut samples = Vec::new();
        for (index, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Some((input, output)) = line.split_once('|') else {
                return Err(DatasetReadError::Line {
                    line: index + 1,
                    message: "missing '|' separator.".into(),
                });
            };
            samples.push(Sample {
                input: input.trim().into(),
                output: output.trim().into(),
            });
        }
        Ok(samples)
    }
    fn read_jsonl(&self) -> Result<Vec<Sample>, DatasetReadError> {
        let content = fs::read_to_string(&self.path)?;
        let mut samples = Vec::new();
        for (index, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let value: Value = serde_json::from_str(line).map_err(|_| DatasetReadError::Line {
                line: index + 1,
                message: "invalid JSON.".into(),
            })?;
            samples.push(
                parse_sample(&value).map_err(|message| DatasetReadError::Line {
                    line: index + 1,
                    message,
                })?,
            );
        }
        Ok(samples)
    }
    fn read_json(&self) -> Result<Vec<Sample>, DatasetReadError> {
        let content = fs::read_to_string(&self.path)?;
        let value: Value = serde_json::from_str(&content)?;
        let items = value.as_array().ok_or_else(|| DatasetReadError::Line {
            line: 1,
            message: "JSON dataset must be a list.".into(),
        })?;
        items
            .iter()
            .enumerate()
            .map(|(index, item)| {
                parse_sample(item).map_err(|message| DatasetReadError::Line {
                    line: index + 1,
                    message,
                })
            })
            .collect()
    }
}
fn parse_sample(value: &Value) -> Result<Sample, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "item must be an object.".to_string())?;
    let input = object
        .get("input")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing input.".to_string())?;
    let output = object
        .get("output")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing output.".to_string())?;
    Ok(Sample {
        input: input.to_owned(),
        output: output.to_owned(),
    })
}
