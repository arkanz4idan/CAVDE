use burn::tensor::Device;
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use cavde::cavde::{
    checkpoint::Checkpoint,
    config::Config,
    data::{
        dataloader::DataLoader, dataset::ConversationDataset, reader::DatasetReader,
        statistics::DatasetStatistics,
    },
    engine::TrainingEngine,
    factory::create_model,
    scheduler::LearningRateScheduler,
    tokenizers::SimpleTokenizer,
    trainer::Trainer,
    vocabulary::Vocabulary,
};

const SUPPORTED: &[&str] = &["json", "jsonl", "txt"];
const IGNORED: &[&str] = &[
    "vocab",
    "tokenizer",
    "config",
    "checkpoint",
    "model",
    "pyfolder",
];

fn scan_datasets(directory: &Path) -> anyhow::Result<Vec<PathBuf>> {
    if !directory.exists() {
        return Ok(Vec::new());
    }
    let mut files = Vec::new();
    let mut pending = vec![directory.to_path_buf()];
    while let Some(path) = pending.pop() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
                continue;
            }
            let extension = path
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            let full_path = path.to_string_lossy().to_ascii_lowercase();
            if SUPPORTED.contains(&extension.as_str())
                && !IGNORED.iter().any(|word| full_path.contains(word))
            {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

pub fn run(device: Device) -> anyhow::Result<()> {
    let config_path = Path::new("config.yaml");
    let config = if config_path.exists() {
        Config::load(config_path)?
    } else {
        Config::default()
    };
    std::fs::create_dir_all(&config.checkpoint_dir)?;

    println!("======================================================");
    println!("                  CAVDE TRAINER");
    println!("======================================================");
    println!("[1/5] Scanning datasets...");
    let dataset_files = scan_datasets(&config.dataset_dir)?;
    anyhow::ensure!(
        !dataset_files.is_empty(),
        "No supported dataset files found in {}",
        config.dataset_dir.display()
    );

    let mut raw_samples = Vec::new();
    let mut vocabulary = Vocabulary::new();
    let mut statistics = DatasetStatistics::default();
    for path in &dataset_files {
        println!("Dataset: {}", path.display());
        let samples = DatasetReader::new(path).read_all()?;
        for sample in &samples {
            statistics.update(sample);
        }
        for sample in &samples {
            vocabulary.add_text(&sample.input);
            vocabulary.add_text(&sample.output);
        }
        raw_samples.extend(samples);
    }
    statistics.report();
    vocabulary.build();
    anyhow::ensure!(
        vocabulary.len() > 5,
        "Dataset did not produce any vocabulary tokens"
    );
    vocabulary.save(config.checkpoint_dir.join("vocab.json"))?;
    println!(
        "Loaded {} files and {} samples; vocabulary={}",
        dataset_files.len(),
        raw_samples.len(),
        vocabulary.len()
    );

    let tokenizer = SimpleTokenizer::new(vocabulary);
    let dataset = ConversationDataset::from_samples(&raw_samples, &tokenizer);
    let mut dataloader = DataLoader::new(&dataset, config.batch_size, true, config.seed);
    anyhow::ensure!(
        !dataloader.is_empty(),
        "Dataset contains no trainable batches"
    );

    println!("[2/5] Creating model...");
    let training_device = device.autodiff();
    let model = create_model(tokenizer.vocab.len(), &config, &training_device);
    println!("Model parameters: {}", model.parameter_count());

    println!("[3/5] Creating training engine...");
    let mut engine = TrainingEngine::new(
        model,
        training_device,
        config.learning_rate,
        config.weight_decay as f32,
    );

    println!("[4/5] Creating scheduler and checkpoint manager...");
    let mut scheduler = LearningRateScheduler::new(
        config.scheduler.clone(),
        config.learning_rate,
        config.step_size,
        config.gamma,
        config.t_max,
    );
    let checkpoint = Checkpoint::new(&config.checkpoint_dir)?;
    let start_epoch = if checkpoint.has_last() {
        let metadata = checkpoint.restore_last(&mut engine)?;
        let epoch = metadata["epoch"].as_u64().unwrap_or(0) as usize;
        scheduler.set_step(epoch);
        println!("Restored model and optimizer from epoch {epoch}.");
        epoch
    } else {
        0
    };

    println!("[5/5] Starting training...");
    let started = Instant::now();
    let mut trainer = Trainer::new(config.epochs)
        .with_start_epoch(start_epoch)
        .with_validation();
    if let Ok(best) = checkpoint.metadata("best") {
        trainer.best_loss = best["loss"].as_f64().unwrap_or(f64::INFINITY);
    }
    let reports = trainer.train(&mut engine, &mut dataloader, &mut scheduler, &checkpoint)?;
    println!(
        "Training finished: {} epochs in {:.1}s",
        reports.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
