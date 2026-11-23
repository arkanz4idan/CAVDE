#[derive(Debug, Clone)]
pub struct LearningRateScheduler {
    kind: String,
    step_size: usize,
    gamma: f64,
    t_max: usize,
    base_lr: f64,
    step: usize,
}
impl LearningRateScheduler {
    pub fn new(
        kind: impl Into<String>,
        base_lr: f64,
        step_size: usize,
        gamma: f64,
        t_max: usize,
    ) -> Self {
        Self {
            kind: kind.into().to_lowercase(),
            step_size,
            gamma,
            t_max,
            base_lr,
            step: 0,
        }
    }
    pub fn current_lr(&self) -> f64 {
        match self.kind.as_str() {
            "step" => self.base_lr * self.gamma.powi((self.step / self.step_size.max(1)) as i32),
            "cosine" => {
                self.base_lr
                    * 0.5
                    * (1.0
                        + (std::f64::consts::PI * self.step as f64 / self.t_max.max(1) as f64)
                            .cos())
            }
            _ => self.base_lr,
        }
    }
    pub fn set_step(&mut self, step: usize) {
        self.step = step;
    }
    pub fn step(&mut self) {
        self.step += 1;
    }
}
