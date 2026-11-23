use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};
pub struct CavdeLogger {
    name: String,
    path: PathBuf,
    file: Mutex<File>,
}
impl CavdeLogger {
    fn write(&self, level: &str, message: &str) -> anyhow::Result<()> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let mut file = self
            .file
            .lock()
            .map_err(|_| anyhow::anyhow!("logger lock poisoned"))?;
        writeln!(file, "[{timestamp}] {level} : {message}")?;
        file.flush()?;
        Ok(())
    }
    pub fn info(&self, message: impl AsRef<str>) -> anyhow::Result<()> {
        self.write("INFO", message.as_ref())
    }
    pub fn warning(&self, message: impl AsRef<str>) -> anyhow::Result<()> {
        self.write("WARNING", message.as_ref())
    }
    pub fn error(&self, message: impl AsRef<str>) -> anyhow::Result<()> {
        self.write("ERROR", message.as_ref())
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
}
pub fn create_logger(directory: &Path) -> anyhow::Result<CavdeLogger> {
    create_logger_named(directory, "CAVDE")
}
pub fn create_logger_named(
    directory: &Path,
    name: impl Into<String>,
) -> anyhow::Result<CavdeLogger> {
    fs::create_dir_all(directory)?;
    let path = directory.join("train.log");
    let file = OpenOptions::new().create(true).append(true).open(&path)?;
    Ok(CavdeLogger {
        name: name.into(),
        path,
        file: Mutex::new(file),
    })
}
