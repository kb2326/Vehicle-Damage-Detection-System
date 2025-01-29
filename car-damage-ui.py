import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from car_damage_detector import CarDamageDetector  # Import from previous file

def load_and_prep_image(uploaded_file):
    """
    Load and prepare the uploaded image
    """
    # Read image
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB (in case it's RGBA)
    image = image.convert('RGB')
    
    # Resize to 640x640 while maintaining aspect ratio
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    
    if aspect_ratio > 1:
        new_width = 640
        new_height = int(640 / aspect_ratio)
    else:
        new_height = 640
        new_width = int(640 * aspect_ratio)
        
    image = image.resize((new_width, new_height))
    
    # Convert to numpy array for OpenCV
    return np.array(image)

def main():
    st.set_page_config(page_title="Car Damage Detection", layout="wide")
    
    # Header
    st.title("Car Damage Detection System")
    
    # Photography Guidelines
    st.header("📸 Photo Guidelines")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Do's:
        - Take photos from 10-15 feet away
        - Ensure good lighting conditions
        - Capture the entire car section
        - Use landscape orientation
        - Keep the camera steady
        """)
    
    with col2:
        st.markdown("""
        ### Don'ts:
        - Don't take extremely close-up shots
        - Avoid poor lighting or shadows
        - Don't take blurry photos
        - Avoid extreme angles
        - Don't crop the damaged area
        """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image (recommended size: 640x640)", 
                                   type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        try:
            # Load and process image
            image = load_and_prep_image(uploaded_file)
            
            # Display original image
            st.subheader("Original Image")
            st.image(image, channels="RGB", use_column_width=True)
            
            # Initialize detector
            detector = CarDamageDetector("Weights/best.pt")
            
            # Process image
            with st.spinner('Detecting damage...'):
                processed_img, detections = detector.detect_damage(image)
            
            # Display results
            st.subheader("Detection Results")
            st.image(processed_img, channels="RGB", use_column_width=True)
            
            # Display detections in a table
            if detections:
                st.subheader("Detected Damages")
                
                # Create a dataframe for better visualization
                damage_data = {
                    'Damage Type': [d['class'] for d in detections],
                    'Confidence': [f"{d['confidence']:.2%}" for d in detections]
                }
                
                st.table(damage_data)
                
                # Add severity estimation
                st.subheader("Damage Assessment")
                total_damages = len(detections)
                avg_confidence = sum(d['confidence'] for d in detections) / total_damages
                
                if total_damages >= 3:
                    severity = "High"
                    color = "🔴"
                elif total_damages == 2:
                    severity = "Medium"
                    color = "🟡"
                else:
                    severity = "Low"
                    color = "🟢"
                
                st.markdown(f"""
                ### Overall Assessment:
                - Number of damages detected: **{total_damages}**
                - Average confidence score: **{avg_confidence:.2%}**
                - Estimated severity: {color} **{severity}**
                """)
            else:
                st.info("No damage detected in the image.")
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Please try uploading a different image.")
    
    # Add usage instructions at the bottom
    st.markdown("""
    ---
    ### How to use:
    1. Review the photo guidelines above
    2. Upload a photo of the car damage
    3. Wait for the system to process the image
    4. Review the detected damages and assessment
    
    Note: For best results, ensure your photos follow the guidelines and are well-lit.
    """)

if __name__ == "__main__":
    main()