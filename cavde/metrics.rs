pub fn token_accuracy(predictions: &[i64], targets: &[i64]) -> f64 {
    if targets.is_empty() {
        return 0.0;
    }
    let correct = predictions
        .iter()
        .zip(targets)
        .filter(|(a, b)| a == b)
        .count();
    correct as f64 / targets.len() as f64
}
pub fn token_accuracy_from_logits(
    logits: &[f32],
    targets: &[i64],
    vocab_size: usize,
) -> anyhow::Result<f64> {
    anyhow::ensure!(vocab_size > 0, "vocab_size must be greater than zero");
    anyhow::ensure!(
        logits.len() == targets.len() * vocab_size,
        "logits shape does not match target count and vocabulary size"
    );
    if targets.is_empty() {
        return Ok(0.0);
    }
    let predictions = logits.chunks_exact(vocab_size).map(|row| {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index as i64)
            .unwrap_or(0)
    });
    let correct = predictions
        .zip(targets.iter())
        .filter(|(pred, target)| *pred == **target)
        .count();
    Ok(correct as f64 / targets.len() as f64)
}
