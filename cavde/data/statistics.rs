use super::reader::Sample;
use std::collections::HashSet;

#[derive(Debug, Clone, Default)]
pub struct DatasetStatistics {
    pub samples: usize,
    input_words: HashSet<String>,
    output_words: HashSet<String>,
}
impl DatasetStatistics {
    pub fn update(&mut self, sample: &Sample) {
        self.samples += 1;
        self.input_words
            .extend(sample.input.split_whitespace().map(str::to_owned));
        self.output_words
            .extend(sample.output.split_whitespace().map(str::to_owned));
    }
    pub fn unique_input_words(&self) -> usize {
        self.input_words.len()
    }
    pub fn unique_output_words(&self) -> usize {
        self.output_words.len()
    }
    pub fn report(&self) {
        println!("\n=============================================");
        println!("Dataset Statistics");
        println!("=============================================");
        println!("Samples : {}", self.samples);
        println!("Unique Input Words : {}", self.unique_input_words());
        println!("Unique Output Words : {}", self.unique_output_words());
    }
}
