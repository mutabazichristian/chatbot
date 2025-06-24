import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model/Tokenizer Loading ---
MODEL_PATH = "./models"

try:
    print("Loading tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✅ Model and tokenizer loaded successfully!")

except Exception as e:
    print(f"❌ Failed to load model: {e}")
    # Fallback to base GPT-2
    print("Loading base GPT-2 as fallback...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token


# --- FIXED Response Generation ---
def generate_response(input_text):
    try:
        # Format input properly for movie recommendations
        prompt = f"USER: {input_text} ASSISTANT:"

        # Tokenize
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate with better parameters
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 50,  # Allow longer responses
                temperature=0.8,  # More creative
                do_sample=True,  # Enable sampling
                top_p=0.9,  # Nucleus sampling
                top_k=50,  # Top-k sampling
                repetition_penalty=1.3,  # Prevent repetition
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,  # Prevent 3-gram repetition
                early_stopping=True,
            )

        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the assistant's response
        if "ASSISTANT:" in full_response:
            response = full_response.split("ASSISTANT:")[-1].strip()
        else:
            response = full_response.replace(prompt, "").strip()

        # Fallback for empty or bad responses
        if not response or len(response) < 10 or "!" * 5 in response:
            response = get_fallback_response(input_text)

        return response

    except Exception as e:
        print(f"Generation error: {e}")
        return get_fallback_response(input_text)


def get_fallback_response(input_text):
    """Provide rule-based responses if model fails"""
    input_lower = input_text.lower()

    if any(word in input_lower for word in ["action", "explosive", "fight"]):
        return "For action movies, I'd recommend Mad Max: Fury Road, John Wick, or Mission Impossible series!"
    elif any(
        word in input_lower
        for word in ["sci-fi", "science", "space", "future"]
    ):
        return "Great sci-fi picks include Blade Runner 2049, Interstellar, and Arrival!"
    elif any(word in input_lower for word in ["comedy", "funny", "laugh"]):
        return "For laughs, try The Grand Budapest Hotel, Knives Out, or Brooklyn Nine-Nine!"
    elif any(word in input_lower for word in ["horror", "scary", "thriller"]):
        return "If you want scares, check out Hereditary, The Conjuring, or Get Out!"
    elif any(word in input_lower for word in ["inception", "nolan", "mind"]):
        return "If you loved Inception, you'd enjoy Memento, The Prestige, and Shutter Island!"
    else:
        return "I'd be happy to recommend movies! What genre or specific films do you enjoy?"


# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🎬 Sidekick - Your Movie Recommendation Assistant")
    gr.Markdown(
        "Ask me about movies! Try: 'I liked Inception' or 'Recommend action movies'"
    )

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=400, label="Movie Chat")
        with gr.Column(scale=1):
            gr.Markdown("### Popular Genres")
            gr.Markdown(
                "- Action & Adventure\n- Sci-Fi & Fantasy\n- Comedy\n- Horror & Thriller\n- Drama\n- Romance"
            )

    with gr.Row():
        msg = gr.Textbox(
            label="Your Message",
            placeholder="What kind of movies do you like?",
            scale=4,
        )
        send_btn = gr.Button("Send", scale=1)
        clear = gr.Button("Clear Chat", scale=1)

    def respond(message, chat_history):
        if not message.strip():
            return chat_history, ""

        bot_message = generate_response(message)
        chat_history.append((message, bot_message))
        return chat_history, ""

    def clear_chat():
        return [], ""

    # Event handlers
    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    send_btn.click(respond, [msg, chatbot], [chatbot, msg])
    clear.click(clear_chat, None, [chatbot, msg])

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
