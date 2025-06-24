from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained(
    "gpt2", pad_token_id=tokenizer.eos_token_id
)

# Load processed data
dataset = tf.data.experimental.load(
    "processed_data",
    element_spec={
        "input_ids": tf.TensorSpec(shape=(None, 128), dtype=tf.int32),
        "attention_mask": tf.TensorSpec(shape=(None, 128), dtype=tf.int32),
        "labels": tf.TensorSpec(shape=(None, 128), dtype=tf.int32),
    },
)

# Configure training
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss)

# Train for 1 epoch (quick demo)
history = model.fit(dataset, epochs=1)

# Save model
model.save_pretrained("sidekick_model")
tokenizer.save_pretrained("sidekick_model")
print("Training complete! Model saved")
