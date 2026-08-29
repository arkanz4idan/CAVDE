from cavde.data.reader import DatasetReader

reader = DatasetReader("datasets/chat.jsonl")

for sample in reader:
    print(sample)