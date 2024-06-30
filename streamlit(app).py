import streamlit as st
import tensorflow as tf
import keras
import sentencepiece as spm
import numpy as np
import os


encoder_model_path = os.path.abspath("essay_sp.model")

@st.cache_resource
def load_model():
    model_path = os.path.abspath("model.keras")
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error(f"Model file not found at {model_path}")
        return None

# Load the model at app startup
model = load_model()


# Initaliazing and loading Encoder Model
sp = spm.SentencePieceProcessor()
sp.load(encoder_model_path)

# Custom CSS
st.markdown("""
<style>
    # body {
        background-color: #00A36C; 
    }
    .stApp {
        background-color: #E4DAC7;  
    }
    .title {
        color: #EF4444;  /* Dark blue color for the title */
        font-size: 36px;
        font-weight: bold;
        
        margin-bottom: 20px;
            
    }
    .stTextInput > div > div > input {
        font-size: 18px;
        padding: 12px 20px;
        border-radius: 10px;
        border: 2px solid #81d4fa;
        background-color: #e0f7fa;
        width: 200%;  /* Make text area wider */
    }
    .stTextInput > div > div > input:focus {
        box-shadow: none;  /* Remove red boundary on focus */
        border-color: #4CAF50;
    }
    .stButton {
        display: flex;
        justify-content: center;  /* Center the button */
    }
    .stButton > button {
        font-size: 18px;
        padding: 12px 24px;
        border-radius: 25px;
        background-color: #388e3c;
        color: white;  /* Red color for submit text */
        border: none;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .score-display {
        font-size: 25px;  /* Smaller score display */
        font-weight: bold;
        color: red;  /* Red color for score */
        background: transparent;
        padding: 10px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        width: 200px;  /* Adjust width as needed */
        margin-left: auto;
        margin-right: auto;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown("<h1 class='title'>Essay Scoring Prediction App</h1>", unsafe_allow_html=True)
st.write( "Maximum marks = 5")


#Encoding Model_Input

def encode_input(text,encoder_model,max_len=500):
    text=encoder_model.encode(text)
    if len(text)>=max_len:
            text=text[:max_len]
    if len(text)<max_len:
        text=np.pad(text,(0,max_len-len(text)))
    return np.array(text)


# Session Value Key
if 'text_input' not in st.session_state:
    st.session_state.text_input = ''
     
# Input box
user_input = st.text_area(" ",height=300,placeholder="Enter Your Text Here And Press Ctrl+Enter to evaluate",value=st.session_state.text_input)


score_placeholder = st.markdown(f"<div class='score-display'>Your score: {0}</div>", unsafe_allow_html=True)



# Submit button
if st.button("Submit") and user_input:
    score = int(np.round(model.predict(encode_input(user_input,sp,max_len=500).reshape(1,500))*6))
    score_placeholder.markdown(f"<div class='score-display'>Your score: {score}</div>", unsafe_allow_html=True)
# Update session state with the current input to ensure it's saved
st.session_state.text_input = user_input

# Clear button

clear_button = st.button("Clear")
if clear_button:
    st.session_state.text_input = " " 
    st.experimental_rerun()










