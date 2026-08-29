from cavde.data.reader import DatasetReader
from cavde.data.validator import DatasetValidator

reader = DatasetReader("datasets/chat.txt")

samples = list(reader)

validator = DatasetValidator()

valid, errors = validator.validate(samples)

print("Valid Samples :", len(valid))
print("Errors :", len(errors))

for error in errors:
    print(error)