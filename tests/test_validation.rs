use cavde::cavde::validation::Validator;
#[test]
fn average_loss_handles_values_and_empty_input() {
    assert_eq!(Validator::average_loss(&[2.0, 4.0]), 3.0);
    assert_eq!(Validator::average_loss(&[]), 0.0);
}
