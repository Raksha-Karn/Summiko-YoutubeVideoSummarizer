from datasets import load_dataset


dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
print(dataset)

print(dataset["train"][0]["article"])
print(dataset["train"][0]["highlights"])

dataset.save_to_disk("cnn_dailymail")
