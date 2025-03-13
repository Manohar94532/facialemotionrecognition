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

# Emotion emojis
emotion_emojis = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😠",
    "fear": "😨",
    "surprise": "😲",
    "neutral": "😐",
    "disgust": "🤢"
}

def main():
    st.set_page_config(
        page_title="Facial Emotion Recognition",
        page_icon="😀",
        layout="wide"
    )
    
    st.title("Facial Emotion Recognition")
    st.write("Upload an image, and the model will predict the detected emotion(s).")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            # Image size adjustment
            st.sidebar.header("Image Settings")
            original_width, original_height = image.size
            aspect_ratio = original_width / original_height
            
            # Default display width is 50% of original, with min 200px and max original size
            default_width = min(original_width, max(200, int(original_width * 0.5)))
            
            # Image size slider
            display_width = st.sidebar.slider(
                "Image Width", 
                min_value=200, 
                max_value=original_width, 
                value=default_width,
                step=50
            )
            
            # Increment/decrement buttons
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("➖ Smaller"):
                    display_width = max(200, display_width - 50)
            with col2:
                if st.button("➕ Larger"):
                    display_width = min(original_width, display_width + 50)
            
            # Calculate height based on aspect ratio
            display_height = int(display_width / aspect_ratio)
            
            # Display resized image
            resized_image = image.resize((display_width, display_height), Image.LANCZOS)
            st.image(resized_image, caption="Uploaded Image", use_column_width=False)
            
            with st.spinner("Processing..."):
                # Convert PIL image to cv2 format (use original image for processing)
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
                    dominant_emotion_lower = dominant_emotion.lower()
                    
                    # Convert emotion scores to DataFrame
                    df = pd.DataFrame(list(emotions_dict.items()), columns=['Emotion', 'Score'])
                    df['Score'] = df['Score'] * 100  # Convert to percentage
                    df['Score'] = df['Score'].round(2)
                    df = df.sort_values(by='Score', ascending=False)
                    
                    # Create emoji display
                    emoji = emotion_emojis.get(dominant_emotion_lower, "❓")
                    
                    # Display emotion header with emoji
                    st.markdown(
                        f"<h2 style='text-align: center;'>"
                        f"Detected Emotion: {dominant_emotion.capitalize()} {emoji}</h2>", 
                        unsafe_allow_html=True
                    )
                    
                    # Display avatar with detected emotion
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # Display bar chart with emotions
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
                        if dominant_emotion_lower in emotion_animations:
                            lottie_animation = load_lottieurl(emotion_animations[dominant_emotion_lower])
                            if lottie_animation:
                                st_lottie(lottie_animation, height=300)
                            else:
                                st.write("Animation failed to load.")
                        else:
                            st.info(f"No animation available for {dominant_emotion}")
                    
                    # Display scores in a clean table
                    st.write("### Detailed Scores")
                    st.dataframe(
                        df.style.format({'Score': '{:.2f}%'}), 
                        use_container_width=True
                    )
                    
                    # Display face locations
                    if st.checkbox("Show detected face", value=True):
                        face_locations = emotions[0]["box"]
                        x, y, w, h = face_locations
                        
                        # Draw rectangle on image
                        img_with_face = np.array(image).copy()
                        cv2.rectangle(img_with_face, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Resize the image with bounding box
                        img_with_face_pil = Image.fromarray(cv2.cvtColor(img_with_face, cv2.COLOR_BGR2RGB))
                        resized_img_with_face = img_with_face_pil.resize((display_width, display_height), Image.LANCZOS)
                        
                        # Display image with bounding box
                        st.image(resized_img_with_face, caption="Detected Face", use_column_width=False)
                        
                else:
                    st.warning("No faces detected in the image. Please try another image.")
                
        except Exception as e:
            st.error(f"Error in emotion detection: {str(e)}")
            st.info("Make sure the image contains a clearly visible face.")

if __name__ == "__main__":
    main()
