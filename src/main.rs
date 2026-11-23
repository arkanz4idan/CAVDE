use anyhow::Result;
use cavde::backend::{BackendKind, create_device};
#[path = "../benchmark.rs"]
mod benchmark;
#[path = "../chat.rs"]
mod chat;
#[path = "../evaluate.rs"]
mod evaluate;
#[path = "../export.rs"]
mod export;
#[path = "../train.rs"]
mod train;
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum BackendArg {
    Wgpu,
    Cuda,
    Flex,
}
impl From<BackendArg> for BackendKind {
    fn from(value: BackendArg) -> Self {
        match value {
            BackendArg::Wgpu => Self::Wgpu,
            BackendArg::Cuda => Self::Cuda,
            BackendArg::Flex => Self::Flex,
        }
    }
}
#[derive(Debug, Parser)]
#[command(name = "CAVDE", about = "CAVDE AI Framework")]
struct Cli {
    #[arg(long, value_enum, global = true, default_value = "wgpu")]
    backend: BackendArg,
    #[command(subcommand)]
    command: Command,
}
#[derive(Debug, Subcommand)]
enum Command {
    Train,
    Chat,
    Benchmark,
    Evaluate,
    Export,
}
fn main() -> Result<()> {
    let cli = Cli::parse();
    let device = create_device(cli.backend.into())?;
    match cli.command {
        Command::Train => train::run(device),
        Command::Chat => chat::run(device),
        Command::Benchmark => benchmark::run(device),
        Command::Evaluate => evaluate::run(device),
        Command::Export => export::run(),
    }
}
