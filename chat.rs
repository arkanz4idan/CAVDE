use std::io::{self, Write};

use burn::tensor::Device;

use cavde::cavde::{
    checkpoint::Checkpoint, config::Config, factory::create_model, generation::GreedyGenerator,
    tokenizers::SimpleTokenizer, vocabulary::Vocabulary,
};

pub fn run(device: Device) -> anyhow::Result<()> {
    let config = Config::load("config.yaml").unwrap_or_default();
    println!("==================================================");
    println!("                  CAVDE CHAT");
    println!("==================================================");

    let vocab_path = config.checkpoint_dir.join("vocab.json");
    let mut vocabulary = Vocabulary::new();
    if vocab_path.exists() {
        vocabulary.load(&vocab_path)?;
        println!("Vocabulary loaded: {} tokens", vocabulary.len());
    } else {
        let tokenizer = cavde::cavde::factory::create_tokenizer();
        vocabulary = tokenizer.vocab;
        println!(
            "No saved vocabulary found; using dataset vocabulary ({} tokens).",
            vocabulary.len()
        );
    }

    anyhow::ensure!(
        vocabulary.len() >= 5,
        "Vocabulary is empty; train or prepare a dataset first."
    );
    let tokenizer = SimpleTokenizer::new(vocabulary);
    let model = create_model(tokenizer.vocab.len(), &config, &device);
    let checkpoint = Checkpoint::new(&config.checkpoint_dir)?;
    let model = if checkpoint.directory.join("last.mpk").exists() {
        match checkpoint.load_model(model.clone(), "last.mpk") {
            Ok(loaded) => {
                if let Ok(metadata) = checkpoint.last_metadata() {
                    println!(
                        "Checkpoint loaded: epoch {}, loss {:.6}",
                        metadata["epoch"].as_u64().unwrap_or(0),
                        metadata["loss"].as_f64().unwrap_or(f64::NAN)
                    );
                } else {
                    println!("Checkpoint loaded.");
                }
                loaded
            }
            Err(error) => {
                println!("Warning: could not load checkpoint: {error}");
                println!("Using a newly initialized model.");
                model
            }
        }
    } else {
        println!("No trained checkpoint found; using an untrained model.");
        model
    };

    let mut generator = GreedyGenerator::new(model, tokenizer, device);
    generator.temperature = 0.8;
    generator.top_k = Some(5);
    generator.top_p = Some(0.9);

    println!("Type 'exit' or 'quit' to leave.");
    let stdin = io::stdin();
    loop {
        print!("You : ");
        io::stdout().flush()?;
        let mut prompt = String::new();
        if stdin.read_line(&mut prompt)? == 0 {
            break;
        }
        let prompt = prompt.trim();
        if prompt.is_empty() {
            continue;
        }
        if matches!(prompt.to_ascii_lowercase().as_str(), "exit" | "quit") {
            break;
        }
        match generator.generate(prompt) {
            Ok(answer) => println!("\nCAVDE : {answer}\n"),
            Err(error) => eprintln!("Generation error: {error}"),
        }
    }
    Ok(())
}
