use cavde::cavde::checkpoint::Checkpoint;
#[test]
fn new_checkpoint_starts_without_last_state() {
    let dir = tempfile::tempdir().unwrap();
    let checkpoint = Checkpoint::new(dir.path()).unwrap();
    assert!(!checkpoint.has_last());
}
