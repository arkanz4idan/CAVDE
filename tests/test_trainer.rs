use cavde::cavde::scheduler::LearningRateScheduler;
#[test]
fn step_scheduler_reduces_learning_rate_at_interval() {
    let mut scheduler = LearningRateScheduler::new("step", 1.0, 2, 0.5, 10);
    assert_eq!(scheduler.current_lr(), 1.0);
    scheduler.step();
    scheduler.step();
    assert!((scheduler.current_lr() - 0.5).abs() < 1e-12);
}
