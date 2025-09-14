
import streamlit as st
import joblib

model = joblib.load("sentiment_model.pkl")
emotion_numbers = {
    "sadness": 0,
    "anger": 1,
    "love": 2,
    "surprise": 3,
    "fear": 4,
    "joy": 5
}

number_to_emot = {v:k for k,v in emotion_numbers.items()}

st.set_page_config(page_title = "Emotional Detector", page_icon= "😀")

st.title("🎭 Emotion Detection App")
st.write("Type your text and find the emotion behind it!")

user_input = st.text_area("Enter your text here:")

if st.button("Analyze Emotion"):
    if user_input.strip():
        tfidf = joblib.load("tf_model.pkl")   
        X = tfidf.transform([user_input])  
        pred = model.predict(X)[0]
        emotion_label = number_to_emot[pred]

        print(emotion_label)

        st.success(f"Detected Emotion: **{emotion_label}**")
    else:
        st.warning("Please enter some text before analyzing.")
