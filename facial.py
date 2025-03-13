import streamlit as st
from PIL import Image
import cv2
import numpy as np
from streamlit_lottie import st_lottie
import requests
import pandas as pd
import plotly.express as px
import os
from fer import FER

def load_lottieurl(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return None
        return response.json()
    except Exception as e:
        st.warning(f"Could not load animation: {e}")
        return None

# Emotion animations
emotion_animations = {
    "happy": "https://assets4.lottiefiles.com/packages/lf20_zw0djhar.json",
    "sad": "https://assets9.lottiefiles.com/packages/lf20_ttvteyvs.json",
    "angry": "https://assets2.lottiefiles.com/packages/lf20_Gc9ZDY.json",
    "fear": "https://assets3.lottiefiles.com/packages/lf20_5pGzPy.json",
    "surprise": "https://assets3.lottiefiles.com/packages/lf20_kvtpp3tw.json",
    "neutral": "https://assets3.lottiefiles.com/packages/lf20_j1adxtyb.json",
    "disgust": "https://assets3.lottiefiles.com/packages/lf20_qz0lj1bd.json"
}

def main():
    st.set_page_config(
        page_title="Facial Emotion Recognition",
        page_icon="😀",
        layout="wide"
    )
    
    st.title("Facial Emotion Recognition")
    st.write("Upload an image, and the model will predict the detected emotion(s).")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with st.spinner("Processing..."):
                # Convert PIL image to cv2 format
                img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Initialize FER detector
                detector = FER()
                
                # Detect emotions
                emotions = detector.detect_emotions(img_array)
                
                if emotions and len(emotions) > 0:
                    # Get emotions from the first face detected
                    emotions_dict = emotions[0]["emotions"]
                    
                    # Get dominant emotion
                    dominant_emotion = max(emotions_dict, key=emotions_dict.get)
                    
                    # Convert emotion scores to DataFrame
                    df = pd.DataFrame(list(emotions_dict.items()), columns=['Emotion', 'Score'])
                    df['Score'] = df['Score'] * 100  # Convert to percentage
                    df['Score'] = df['Score'].round(2)
                    df = df.sort_values(by='Score', ascending=False)
                    
                    # Create two columns for layout
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Display bar chart
                        fig = px.bar(
                            df, 
                            x='Emotion', 
                            y='Score',
                            title='Emotion Scores',
                            color='Score',
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Display animation
                        if dominant_emotion.lower() in emotion_animations:
                            st.markdown(f"### Detected: {dominant_emotion.capitalize()}")
                            lottie_animation = load_lottieurl(emotion_animations[dominant_emotion.lower()])
                            if lottie_animation:
                                st_lottie(lottie_animation, height=300)
                            else:
                                st.write("Animation failed to load.")
                        else:
                            st.markdown(f"### Detected: {dominant_emotion.capitalize()}")
                            st.info("No animation available for this emotion.")
                    
                    # Display scores in a clean table
                    st.write("### Detailed Scores")
                    st.dataframe(
                        df.style.format({'Score': '{:.2f}%'}), 
                        use_container_width=True
                    )
                    
                    # Display face locations
                    if st.checkbox("Show detected face"):
                        face_locations = emotions[0]["box"]
                        x, y, w, h = face_locations
                        
                        # Draw rectangle on image
                        img_with_face = np.array(image).copy()
                        cv2.rectangle(img_with_face, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Display image with bounding box
                        st.image(img_with_face, caption="Detected Face", use_column_width=True)
                        
                else:
                    st.warning("No faces detected in the image. Please try another image.")
                
        except Exception as e:
            st.error(f"Error in emotion detection: {str(e)}")
            st.info("Make sure the image contains a clearly visible face.")

if __name__ == "__main__":
    main()
