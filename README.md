# Sidekick

What are you watching next?

**Mutabazi I. Willy Christian**

Demo: [https://youtu.be/wienwXd5NA0](https://youtu.be/wienwXd5NA0)

GitHub: [https://github.com/mutabazichristian/chatbot](https://github.com/mutabazichristian/chatbot)

Data: [https://redialdata.github.io/website/download](https://redialdata.github.io/website/download)

---

## Problem
Netflix users spend 15-20 minutes just browsing for the perfect show, caught in what researchers call "choice paralysis." Studies show that 10 out of 12 Netflix users experience decision paralysis at least once, leading them to exit the platform without making a selection. This isn't laziness, it's a fundamental human response to overwhelming choice. Users scroll endlessly, afraid to commit to something that might disappoint, trapped between too many options and the fear of missing out on the "perfect" choice.

## What does Sidekick do?
Sidekick is a chatbot that chats with users about their preferences (genre, mood, past favorites), and recommends movies that match their interests. It uses natural conversation to narrow a large set of options into a small, high-quality shortlist—making decisions feel easier and more human.

---

## Overview
Sidekick is a conversational AI chatbot designed to help Netflix users discover movies through natural, engaging conversations. Built on a fine-tuned GPT-2 model using the ReDial dataset, Sidekick can recommend movies, discuss preferences, and provide a personalized experience.

## Features

- Domain-specific movie recommendations based on user preferences
- Conversational discovery of user tastes and interests
- Gradio-powered web interface for easy interaction
- Utilizes a large dataset of real movie recommendation dialogues

## Dataset

The chatbot is trained on over 10,000 movie recommendation conversations from the [ReDial dataset](https://huggingface.co/datasets/redial), which contains authentic dialogues between users seeking and giving movie suggestions.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/sidekick-chatbot.git
   cd sidekick-chatbot
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Chatbot

Start the chatbot server with:

```bash
python app.py
```

Once running, open your browser and navigate to [http://localhost:7860](http://localhost:7860) to interact with Sidekick.

## Project Structure

```
src/
  app.py                # Main application (Gradio interface, chatbot logic)
  preprocess.py         # Data preprocessing scripts
  train.py              # Model training script

data/
  movies_with_mentions.csv   # Movie metadata with mention info
  train_data.jsonl           # Training data (JSONL format)
  test_data.jsonl            # Test data (JSONL format)
```

## Model Performance

| Metric         | Value |
|----------------|-------|
| Training Loss  | 2.18  |


## Citation

the ReDial dataset [ReDial paper](https://arxiv.org/abs/1809.01984).

## License

This project is licensed under the MIT License.
