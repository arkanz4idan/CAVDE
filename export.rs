use std::fs;

use anyhow::{Context, Result};

use cavde::cavde::config::Config;

/// Export the latest native Burn checkpoint to the configured model directory.
pub fn run() -> Result<()> {
    let config = Config::load("config.yaml").unwrap_or_default();
    let source = config.checkpoint_dir.join("last.mpk");
    let destination_dir = std::path::PathBuf::from("models");
    let destination = destination_dir.join("cavde-base.mpk");

    if !source.exists() {
        println!("No Rust checkpoint found at {}.", source.display());
        return Ok(());
    }

    fs::create_dir_all(&destination_dir).with_context(|| {
        format!(
            "could not create export directory {}",
            destination_dir.display()
        )
    })?;
    fs::copy(&source, &destination).with_context(|| {
        format!(
            "could not export {} to {}",
            source.display(),
            destination.display()
        )
    })?;

    println!("==================================================");
    println!("                 CAVDE EXPORT");
    println!("==================================================");
    println!("Source      : {}", source.display());
    println!("Destination : {}", destination.display());
    println!("Export complete.");
    Ok(())
}
