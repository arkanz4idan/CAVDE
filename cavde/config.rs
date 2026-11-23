use serde::Deserialize;
use std::path::{Path, PathBuf};

fn project_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(name)
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default = "default_project_name")]
    pub project_name: String,
    #[serde(default = "default_version")]
    pub version: String,
    #[serde(default = "default_dataset_dir")]
    pub dataset_dir: PathBuf,
    #[serde(default = "default_checkpoint_dir")]
    pub checkpoint_dir: PathBuf,
    #[serde(default = "default_log_dir")]
    pub log_dir: PathBuf,
    #[serde(default = "default_export_dir")]
    pub export_dir: PathBuf,
    #[serde(default = "default_cache_dir")]
    pub cache_dir: PathBuf,
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    #[serde(default = "default_num_layers")]
    pub num_layers: usize,
    #[serde(default = "default_feed_forward")]
    pub feed_forward: usize,
    #[serde(default = "default_dropout")]
    pub dropout: f64,
    #[serde(default = "default_max_sequence_length")]
    pub max_sequence_length: usize,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,
    #[serde(default = "default_gradient_clip")]
    pub gradient_clip: f64,
    #[serde(default = "default_min_word_frequency")]
    pub min_word_frequency: usize,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_save_every_epoch")]
    pub save_every_epoch: bool,
    #[serde(default = "default_save_best_only")]
    pub save_best_only: bool,
    #[serde(default = "default_scheduler")]
    pub scheduler: String,
    #[serde(default = "default_step_size")]
    pub step_size: usize,
    #[serde(default = "default_gamma")]
    pub gamma: f64,
    #[serde(default = "default_t_max")]
    pub t_max: usize,
}

fn default_project_name() -> String {
    "CAVDE".into()
}
fn default_version() -> String {
    "2.0.0-alpha".into()
}
fn default_dataset_dir() -> PathBuf {
    project_path("datasets")
}
fn default_checkpoint_dir() -> PathBuf {
    project_path("checkpoints")
}
fn default_log_dir() -> PathBuf {
    project_path("logs")
}
fn default_export_dir() -> PathBuf {
    project_path("exports")
}
fn default_cache_dir() -> PathBuf {
    project_path("cache")
}
fn default_d_model() -> usize {
    256
}
fn default_num_heads() -> usize {
    8
}
fn default_num_layers() -> usize {
    6
}
fn default_feed_forward() -> usize {
    1024
}
fn default_dropout() -> f64 {
    0.1
}
fn default_max_sequence_length() -> usize {
    256
}
fn default_batch_size() -> usize {
    32
}
fn default_learning_rate() -> f64 {
    3e-4
}
fn default_epochs() -> usize {
    1000
}
fn default_weight_decay() -> f64 {
    0.01
}
fn default_gradient_clip() -> f64 {
    1.0
}
fn default_min_word_frequency() -> usize {
    2
}
fn default_vocab_size() -> usize {
    50000
}
fn default_seed() -> u64 {
    42
}
fn default_save_every_epoch() -> bool {
    true
}
fn default_save_best_only() -> bool {
    true
}
fn default_scheduler() -> String {
    "step".into()
}
fn default_step_size() -> usize {
    5
}
fn default_gamma() -> f64 {
    0.5
}
fn default_t_max() -> usize {
    10
}

impl Default for Config {
    fn default() -> Self {
        Self {
            project_name: default_project_name(),
            version: default_version(),
            dataset_dir: default_dataset_dir(),
            checkpoint_dir: default_checkpoint_dir(),
            log_dir: default_log_dir(),
            export_dir: default_export_dir(),
            cache_dir: default_cache_dir(),
            d_model: default_d_model(),
            num_heads: default_num_heads(),
            num_layers: default_num_layers(),
            feed_forward: default_feed_forward(),
            dropout: default_dropout(),
            max_sequence_length: default_max_sequence_length(),
            batch_size: default_batch_size(),
            learning_rate: default_learning_rate(),
            epochs: default_epochs(),
            weight_decay: default_weight_decay(),
            gradient_clip: default_gradient_clip(),
            min_word_frequency: default_min_word_frequency(),
            vocab_size: default_vocab_size(),
            seed: default_seed(),
            save_every_epoch: default_save_every_epoch(),
            save_best_only: default_save_best_only(),
            scheduler: default_scheduler(),
            step_size: default_step_size(),
            gamma: default_gamma(),
            t_max: default_t_max(),
        }
    }
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_yaml::from_str(&content)?;
        anyhow::ensure!(config.d_model > 0, "d_model must be greater than zero");
        anyhow::ensure!(config.num_heads > 0, "num_heads must be greater than zero");
        anyhow::ensure!(
            config.d_model % config.num_heads == 0,
            "d_model must be divisible by num_heads"
        );
        anyhow::ensure!(
            config.num_layers > 0,
            "num_layers must be greater than zero"
        );
        anyhow::ensure!(
            (0.0..1.0).contains(&config.dropout),
            "dropout must be in [0, 1)"
        );
        Ok(config)
    }
}
