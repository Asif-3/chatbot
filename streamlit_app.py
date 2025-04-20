import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load LLM (you can swap the model below with others from Hugging Face)
@st.cache_resource
def load_model():
    model_name = "tiiuae/falcon-rw-1b"  # small and runs on CPU/GPU
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

st.title("🧠 Open Source ChatGPT")
st.caption("Chatbot using local LLM via Hugging Face Transformers (No API needed)")

# Session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Generate model response
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

# Chat input
if user_input := st.chat_input("Ask me anything..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get and display model response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(user_input)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
