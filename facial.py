import streamlit as st
from PIL import Image
import cv2
from deepface import DeepFace
import numpy as np
from streamlit_lottie import st_lottie
import requests
import pandas as pd
import plotly.express as px

def load_lottieurl(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()

emotion_animations = {
    "happy": "https://assets4.lottiefiles.com/packages/lf20_zw0djhar.json",
    "sad": "https://assets9.lottiefiles.com/packages/lf20_ttvteyvs.json",
    "angry": "https://assets2.lottiefiles.com/packages/lf20_Gc9ZDY.json",
    "fear": "https://assets3.lottiefiles.com/packages/lf20_5pGzPy.json",
    "surprise": "https://assets3.lottiefiles.com/packages/lf20_kvtpp3tw.json",
    "neutral": "https://assets3.lottiefiles.com/packages/lf20_j1adxtyb.json"
}

st.title("Facial Emotion Recognition")
st.write("Upload an image, and the model will predict the detected emotion(s).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    try:
        results = DeepFace.analyze(img_array, actions=["emotion"], enforce_detection=False)
        
        if isinstance(results, list):
            result = results[0]
        else:
            result = results

        emotion = result.get("dominant_emotion", "Unknown").lower()
        emotions_scores = result.get("emotion", {})

        # Convert emotion scores to DataFrame for better visualization
        df = pd.DataFrame(list(emotions_scores.items()), columns=['Emotion', 'Score'])
        df['Score'] = df['Score'].round(2)
        
        # Create two columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Display bar chart
            fig = px.bar(df, x='Emotion', y='Score',
                        title='Emotion Scores',
                        color='Score',
                        color_continuous_scale='Viridis')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Display animation
            if emotion in emotion_animations:
                st.markdown(f"### Detected: {emotion.capitalize()}")
                lottie_animation = load_lottieurl(emotion_animations[emotion])
                if lottie_animation:
                    st_lottie(lottie_animation, height=300)
                else:
                    st.write("Animation failed to load.")

        # Display scores in a clean table
        st.write("### Detailed Scores")
        st.dataframe(df.style.format({'Score': '{:.2f}%'}), use_container_width=True)

    except Exception as e:
        st.write("Error in emotion detection:", str(e))