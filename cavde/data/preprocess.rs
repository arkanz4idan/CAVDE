use std::{fs, path::Path};
use thiserror::Error;
#[derive(Debug, Error)]
pub enum DatasetValidationError {
    #[error("Dataset file not found: {0}")]
    NotFound(String),
    #[error("Unsupported dataset: {0}")]
    Unsupported(String),
    #[error("Dataset read error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid JSON record: {0}")]
    Json(#[from] serde_json::Error),
}
pub const SUPPORTED_EXTENSIONS: [&str; 2] = ["txt", "jsonl"];
pub fn validate_dataset(path: &Path) -> Result<(), DatasetValidationError> {
    if !path.exists() {
        return Err(DatasetValidationError::NotFound(path.display().to_string()));
    }
    let ext = path
        .extension()
        .and_then(|x| x.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if !SUPPORTED_EXTENSIONS.contains(&ext.as_str()) {
        return Err(DatasetValidationError::Unsupported(format!(".{ext}")));
    }
    Ok(())
}
pub fn iter_dataset(path: &Path) -> Result<Vec<serde_json::Value>, DatasetValidationError> {
    validate_dataset(path)?;
    let extension = path
        .extension()
        .and_then(|x| x.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let content = fs::read_to_string(path)?;
    if extension == "txt" {
        let records = content
            .lines()
            .filter_map(|line| {
                let line = line.trim();
                if line.is_empty() {
                    return None;
                }
                let (input, output) = line.split_once('|')?;
                Some(serde_json::json!({"input": input.trim(), "output": output.trim()}))
            })
            .collect();
        return Ok(records);
    }
    content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| Ok(serde_json::from_str(line.trim())?))
        .collect()
}
