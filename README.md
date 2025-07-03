# SlangBot – Powered by Gemma

**SlangBot** is a Gen-Z-flavored chatbot powered by the [Google Gemma-2B-IT](https://huggingface.co/google/gemma-2b-it) model, trained to respond in internet slang, memes, and vibes. It can explain slang, carry on casual conversations, and help you sound *hella lit* online.

---

## Files in this Repo

- `SlangBot.ipynb` – Full notebook showing how the bot works and how the model is loaded and slangified.
- `app.py` – Production-ready Gradio app script for deploying the bot using the Gemma model.
- `requirements.txt` – Dependencies for the app.
- `LICENSE` – Apache License 2.0 for open-source sharing and reuse.

---

## Run Locally

Clone the repo and run the chatbot locally:

```bash
git clone https://github.com/YOUR_USERNAME/SlangBot.git
cd SlangBot
pip install -r requirements.txt
python app.py
```

> Make sure your system supports PyTorch with GPU or has enough RAM to run `gemma-2b-it`.

---

## Model Source

The model weights are hosted [here](https://huggingface.co/pranav2711/SlangBot) and based on `google/gemma-2b-it`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("pranav2711/SlangBot")
model = AutoModelForCausalLM.from_pretrained("pranav2711/SlangBot")
```

---

## Features

* Chat in Gen-Z slang like a influencer.
* Translate regular sentences to meme-speak.
* Explain terms like *sus*, *lowkey*, *skrrt*, etc.
* Runs on Hugging Face Spaces or locally with Gradio.

---

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](./LICENSE) for details.

---

## Contributing

Feel free to fork the repo, create a pull request, or open issues for improvements. PRs that make the bot more cool, are welcome. 

---
