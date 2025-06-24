import json
import pandas as pd
from transformers import GPT2Tokenizer
import tensorflow as tf
import os


# load dataset
def load_dataset(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


train_data = load_dataset("data/train_data.jsonl")
test_data = load_dataset("data/test_data.jsonl")
all_data = train_data + test_data


# Extract conversation pairs
def extract_pairs(convo):
    messages = convo["messages"]
    return [
        (" ".join(msg["text"] for msg in messages[:i]), messages[i]["text"])
        for i in range(1, len(messages))
    ]


pairs = []
for convo in all_data:
    pairs.extend(extract_pairs(convo))

if not pairs:
    print("Warning: No messages found. Using alternative extraction method")
    for convo in all_data:
        if "conversation" in convo:
            messages = convo["conversation"]
            pairs.extend(
                [
                    (" ".join(messages[:i]), messages[i])
                    for i in range(1, len(messages))
                ]
            )

df = pd.DataFrame(pairs, columns=["context", "response"])
# preprocessing
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def tokenize_data(row):
    return tokenizer(
        f"CONTEXT: {row['context']} RESPONSE: {row['response']}<|endoftext|>",
        max_length=128,
        truncation=True,
        padding="max_length",
    )


tokenized_data = df.apply(tokenize_data, axis=1)
input_ids = [x["input_ids"] for x in tokenized_data]
attention_mask = [x["attention_mask"] for x in tokenized_data]

# Create TensorFlow dataset
dataset = (
    tf.data.Dataset.from_tensor_slices(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
        }
    )
    .batch(8)
    .prefetch(tf.data.AUTOTUNE)
)

# Save processed data
tf.data.experimental.save(dataset, "processed_data")
df.to_csv("conversation_pairs.csv", index=False)
print("Preprocessing complete! Processed", len(df), "conversation pairs")
