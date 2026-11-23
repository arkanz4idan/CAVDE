use burn::tensor::{Device, Int, Tensor, TensorData};

use super::{
    nn::CAVDEModel,
    tokenizers::{SimpleTokenizer, base::BaseTokenizer},
};

/// Autoregressive sampler matching the Python GreedyGenerator options.
/// Despite the historical class name, top-k/top-p enable stochastic sampling.
pub struct GreedyGenerator {
    pub model: CAVDEModel,
    pub tokenizer: SimpleTokenizer,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    device: Device,
}

impl GreedyGenerator {
    pub fn new(model: CAVDEModel, tokenizer: SimpleTokenizer, device: Device) -> Self {
        Self {
            model,
            tokenizer,
            max_new_tokens: 32,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            device,
        }
    }

    pub fn generate(&self, prompt: &str) -> anyhow::Result<String> {
        anyhow::ensure!(
            self.temperature.is_finite() && self.temperature > 0.0,
            "temperature must be finite and greater than zero"
        );
        if let Some(k) = self.top_k {
            anyhow::ensure!(k > 0, "top_k must be greater than zero");
        }
        if let Some(p) = self.top_p {
            anyhow::ensure!(
                p.is_finite() && p > 0.0 && p <= 1.0,
                "top_p must be in the interval (0, 1]"
            );
        }

        let bos = *self
            .tokenizer
            .vocab
            .token_to_id
            .get("<BOS>")
            .ok_or_else(|| anyhow::anyhow!("vocabulary is missing <BOS>"))?;
        let eos = *self
            .tokenizer
            .vocab
            .token_to_id
            .get("<EOS>")
            .ok_or_else(|| anyhow::anyhow!("vocabulary is missing <EOS>"))?;
        let mut tokens = Vec::with_capacity(self.max_new_tokens + prompt.len() + 1);
        tokens.push(bos);
        tokens.extend(self.tokenizer.encode(prompt));

        for _ in 0..self.max_new_tokens {
            let input = Tensor::<2, Int>::from_data(
                TensorData::new(tokens.clone(), [1, tokens.len()]),
                &self.device,
            );
            let logits = self.model.forward(input).into_data();
            let values = logits
                .as_slice::<f32>()
                .map_err(|error| anyhow::anyhow!("could not read model logits: {error}"))?;
            let vocab_size = self.tokenizer.vocab.len();
            anyhow::ensure!(
                vocab_size > 0 && values.len() >= vocab_size,
                "model returned an invalid logits tensor"
            );
            let last = &values[values.len() - vocab_size..];
            let next = self.sample_next_token(last)?;
            if next == eos {
                break;
            }
            tokens.push(next);
        }

        Ok(self.tokenizer.decode(&tokens[1..]))
    }

    fn sample_next_token(&self, logits: &[f32]) -> anyhow::Result<i64> {
        let mut ranked: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(index, value)| (index, *value / self.temperature))
            .filter(|(_, value)| value.is_finite())
            .collect();
        anyhow::ensure!(!ranked.is_empty(), "model produced no finite token logits");
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

        if let Some(k) = self.top_k {
            ranked.truncate(k.min(ranked.len()));
        }

        let max_logit = ranked[0].1;
        let mut probabilities: Vec<(usize, f32)> = ranked
            .iter()
            .map(|(index, value)| (*index, (*value - max_logit).exp()))
            .collect();
        let sum: f32 = probabilities
            .iter()
            .map(|(_, probability)| probability)
            .sum();
        anyhow::ensure!(
            sum.is_finite() && sum > 0.0,
            "could not normalize token probabilities"
        );
        for (_, probability) in &mut probabilities {
            *probability /= sum;
        }

        if let Some(top_p) = self.top_p {
            let mut cumulative = 0.0;
            let keep = probabilities
                .iter()
                .position(|(_, probability)| {
                    cumulative += *probability;
                    cumulative >= top_p
                })
                .map(|index| index + 1)
                .unwrap_or(probabilities.len());
            probabilities.truncate(keep.max(1));
            let retained: f32 = probabilities
                .iter()
                .map(|(_, probability)| probability)
                .sum();
            for (_, probability) in &mut probabilities {
                *probability /= retained;
            }
        }

        let draw = rand::random::<f32>();
        let mut cumulative = 0.0;
        for (index, probability) in probabilities {
            cumulative += probability;
            if draw <= cumulative {
                return Ok(index as i64);
            }
        }
        Ok(ranked[0].0 as i64)
    }
}
