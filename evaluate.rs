use burn::tensor::Device;

use cavde::cavde::{
    checkpoint::Checkpoint,
    config::Config,
    data::{dataloader::DataLoader, dataset::ConversationDataset, reader::DatasetReader},
    engine::TrainingEngine,
    factory::create_model,
    tokenizers::SimpleTokenizer,
    validator::ValidationEngine,
    vocabulary::Vocabulary,
};

/// Evaluate the saved model against datasets/chat.jsonl, matching evaluate.py.
pub fn run(device: Device) -> anyhow::Result<()> {
    println!();
    println!("==================================================");
    println!("              CAVDE EVALUATE");
    println!("==================================================");

    let config = Config::load("config.yaml").unwrap_or_default();
    let dataset_path = config.dataset_dir.join("chat.jsonl");
    anyhow::ensure!(
        dataset_path.exists(),
        "Evaluation dataset not found: {}",
        dataset_path.display()
    );

    let samples = DatasetReader::new(&dataset_path).read_all()?;
    anyhow::ensure!(!samples.is_empty(), "Evaluation dataset is empty.");

    let vocab_path = config.checkpoint_dir.join("vocab.json");
    let mut vocabulary = Vocabulary::new();
    if vocab_path.exists() {
        vocabulary.load(&vocab_path)?;
    } else {
        for sample in &samples {
            vocabulary.add_text(&sample.input);
            vocabulary.add_text(&sample.output);
        }
        vocabulary.build();
    }
    anyhow::ensure!(vocabulary.len() >= 5, "Vocabulary is empty.");

    let tokenizer = SimpleTokenizer::new(vocabulary);
    let dataset = ConversationDataset::from_samples(&samples, &tokenizer);
    let mut loader = DataLoader::new(&dataset, 2, false, config.seed);
    anyhow::ensure!(!loader.is_empty(), "Evaluation produced no batches.");

    let checkpoint = Checkpoint::new(&config.checkpoint_dir)?;
    anyhow::ensure!(
        checkpoint.directory.join("last.mpk").exists(),
        "No trained model found."
    );

    let eval_device = device.autodiff();
    let model = create_model(tokenizer.vocab.len(), &config, &eval_device);
    let model = checkpoint.load_model(model, "last.mpk")?;
    let engine = TrainingEngine::new(
        model,
        eval_device,
        config.learning_rate,
        config.weight_decay as f32,
    );

    // Evaluation never calls backward() or optimizer.step().
    let loss = ValidationEngine::new().validate(&engine, &mut loader)?;
    let metadata = checkpoint.last_metadata().ok();

    println!();
    if let Some(state) = metadata {
        println!("Epoch : {}", state["epoch"].as_u64().unwrap_or(0));
    }
    println!("Loss  : {loss:.6}");
    Ok(())
}
