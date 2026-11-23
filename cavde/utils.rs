use rand::{SeedableRng, rngs::StdRng};
use std::path::Path;
pub fn seed_everything(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}
pub fn ensure_directory(path: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(path)?;
    Ok(())
}
pub fn count_parameters<M: burn::module::Module>(model: &M) -> usize {
    model.num_params()
}
pub fn readable_number(number: usize) -> String {
    if number >= 1_000_000 {
        format!("{:.2}M", number as f64 / 1_000_000.0)
    } else if number >= 1_000 {
        format!("{:.2}K", number as f64 / 1_000.0)
    } else {
        number.to_string()
    }
}
