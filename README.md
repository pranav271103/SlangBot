# SlangBot — Conversational Bot using GENz Slangs

SlangBot is a lightweight NLP chatbot that understands informal slang, abbreviations, and internet language. It translates them into clean, formal English using a fine-tuned language model.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/pranav271103/SlangBot.git
cd SlangBot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

---

## `requirements.txt`

```
transformers
torch
streamlit
```

---

## `SlangBot.ipynb` - How I Trained It

This notebook contains the full pipeline to fine-tune a Hugging Face language model using:

* A custom **slang-to-formal** dataset
* Tokenization using `AutoTokenizer` or your **own tokenizer**
* Optional: Slang2Vec embeddings (to semantically embed slang words before generation)

You can reproduce the training using:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
```

Training data is organized as:

```json
{
  "input": "wyd tmrw?",
  "output": "What are you doing tomorrow?"
}
```

I used this structure to fine-tune a causal language model (`GPT2`, `Gemma`, etc.) with Hugging Face `Trainer`.

---

## `app.py` Overview

The chatbot is built using Streamlit. It loads the tokenizer and model directly from your Hugging Face Hub:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("pranav2711/SlangBot")
model = AutoModelForCausalLM.from_pretrained("pranav2711/SlangBot")
```

Input is processed as a prompt → tokenized → passed to model → generated output is decoded and displayed.

---

## License

This project is licensed under the **Apache License 2.0**.

See the full license [here](https://github.com/pranav271103/SlangBot/blob/main/LICENSE) or in the `LICENSE` file.

---

## Technical Highlights

* **Model**: Fine-tuned transformer (e.g. Gemma or GPT-2) for sequence-to-sequence inference.
* **Tokenization**: Trained on slang–formal pairs with optional domain-specific vocabulary.
* **Slang2Vec**: Vector embeddings optionally used during training to enhance slang understanding.
* **Inference**: Text generation using `generate()` with `max_new_tokens`, decoding using `skip_special_tokens=True`.
* **Interface**: Streamlit-based minimal UI that runs completely locally with HF hosting.

---

## Author

**Pranav2711**  
Hugging Face: [https://huggingface.co/pranav2711](https://huggingface.co/pranav2711)

---

## Example Inputs

| Slang Input       | SlangBot Output                        |
| ----------------- | -------------------------------------- |
| `brb gotta dip`   | Be right back, I have to leave.        |
| `wyd tmrw`        | What are you doing tomorrow?           |
| `lol that's wild` | That is very surprising or unexpected. |

---

Feel free to fork, improve, or experiment with SlangBot. Pull requests welcome!
