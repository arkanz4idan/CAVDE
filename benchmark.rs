use std::time::Instant;

use burn::tensor::Device;

use cavde::cavde::{
    checkpoint::Checkpoint, config::Config, factory::create_model, generation::GreedyGenerator,
    tokenizers::SimpleTokenizer, vocabulary::Vocabulary,
};

pub fn run(device: Device) -> anyhow::Result<()> {
    let config = Config::load("config.yaml").unwrap_or_default();
    let vocab_path = config.checkpoint_dir.join("vocab.json");
    let mut vocab = Vocabulary::new();
    if vocab_path.exists() {
        vocab.load(&vocab_path)?;
    } else {
        vocab = cavde::cavde::factory::create_tokenizer().vocab;
    }
    anyhow::ensure!(
        vocab.len() >= 5,
        "Vocabulary is empty; prepare a dataset first."
    );

    let tokenizer = SimpleTokenizer::new(vocab);
    let model = create_model(tokenizer.vocab.len(), &config, &device);
    let checkpoint = Checkpoint::new(&config.checkpoint_dir)?;
    let (model, metadata) = if checkpoint.directory.join("last.mpk").exists() {
        let loaded = checkpoint.load_model(model.clone(), "last.mpk")?;
        (loaded, checkpoint.last_metadata().ok())
    } else {
        (model, None)
    };

    let generator = GreedyGenerator::new(model, tokenizer, device);
    let prompt = "halo";
    let started = Instant::now();
    let response = generator.generate(prompt)?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

    println!("==================================================");
    println!("                CAVDE BENCHMARK");
    println!("==================================================");
    println!("Prompt       : {prompt}");
    println!("Response     : {response}");
    println!("Inference    : {elapsed_ms:.2} ms");
    println!("Vocabulary   : {}", generator.tokenizer.vocab.len());
    println!("Parameters   : {}", generator.model.parameter_count());
    if let Some(state) = metadata {
        println!("Epoch        : {}", state["epoch"].as_u64().unwrap_or(0));
        println!(
            "Loss         : {:.6}",
            state["loss"].as_f64().unwrap_or(f64::NAN)
        );
    }
    Ok(())
}
