from cavde.data.reader import DatasetReader

reader = DatasetReader("datasets/chat.txt")

for sample in reader:
    print(sample)