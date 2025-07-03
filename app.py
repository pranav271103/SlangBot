import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "pranav2711/SlangBot"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return tokenizer, model

def generate_response(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, top_k=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# --- UI ---
st.set_page_config(page_title="SlangBot 🧠", layout="centered")
st.title("🗣️ SlangBot: Your Gen-Z Slang Assistant")

st.markdown("""
Welcome to **SlangBot**, a conversational AI built on the **Google Gemma** model, trained with internet and Gen-Z slang.  
Type anything casual, and SlangBot will slang it up! 😎  
""")

user_input = st.text_input("You:", placeholder="yo dawg, what’s cookin’?", key="input")

if user_input:
    tokenizer, model = load_model()
    with st.spinner("Slangin’... 🔥"):
        response = generate_response(user_input, tokenizer, model)
        st.success(response)
