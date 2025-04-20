import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Use a smaller, more reliable model
model_name = "microsoft/DialoGPT-small"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize chat state
if 'chat_history_ids' not in st.session_state:
    st.session_state.chat_history_ids = None
if 'past_inputs' not in st.session_state:
    st.session_state.past_inputs = []

# App UI
st.title("💬 Chat with Hugging Face Bot")
st.write("Start chatting below:")

# User input
user_input = st.text_input("You:", key="input")

if user_input:
    # Add user input to chat history
    st.session_state.past_inputs.append(user_input)

    # Tokenize user input and append to chat history
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1) if st.session_state.chat_history_ids is not None else new_input_ids

    # Generate bot response
    st.session_state.chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    # Decode and display response
    response = tokenizer.decode(
        st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    # Display conversation
    for i, msg in enumerate(st.session_state.past_inputs):
        st.markdown(f"**You:** {msg}")
        if i == len(st.session_state.past_inputs) - 1:
            st.markdown(f"**Bot:** {response}")
