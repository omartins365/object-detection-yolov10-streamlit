import os
import tempfile

import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLOv10

from config import DEMO_FILES_URL, IMAGE_WIDTH, MODEL_PATH, RESULT_PATH
from utils import download_model


def load_model() -> YOLOv10:
    """ Load the YOLOv10 model. """
    return YOLOv10(MODEL_PATH) if download_model(MODEL_PATH) else None


def process_image(model: YOLOv10, image_path: str, result_path: str) -> bool:
    """ Perform object detection on an image and save the result. """
    try:
        result = model(source=image_path)[0]  # Perform object detection
        result.save(result_path)  # Save the result image
        return True
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return False


def process_video(model: YOLOv10, video_path: str, result_path: str) -> bool:
    """ Perform object detection on a video and save the result. """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Could not open video file.")
            return False

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'vp80')
        out = cv2.VideoWriter(result_path, fourcc, fps, (frame_width, frame_height))
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            results = model(frame)  # Perform object detection on each frame
            out.write(results[0].plot())  # Write processed frame to output video
        return True
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return False
    finally:
        cap.release()
        out.release()
        
     
def display_and_process_file(model: YOLOv10, type_choice: str, temp_path: str, result_path: str) -> None:
    """ Process the uploaded file based on the selected type (Image or Video). """
    try:
        if type_choice == "Image":
            image = Image.open(temp_path)
            st.image(image, "Uploaded Image", width=IMAGE_WIDTH)
            # Process and display result image
            with st.spinner("Processing image..."):
                if process_image(model, temp_path, result_path):
                    st.success(f"Image processed. Result saved: {result_path}")
                    st.image(result_path, "Result Image", width=IMAGE_WIDTH)
                else:
                    st.error("Image processing failed.")
        else:
            st.video(temp_path)
            # Process and display result video
            with st.spinner("Processing video..."):
                result_path = result_path.replace(".mp4", ".webm")
                if process_video(model, temp_path, result_path):
                    st.success(f"Video processed. Result saved: {result_path}")
                    st.video(result_path)
                else:
                    st.error("Video processing failed.")
    except Exception as e:
        st.error(f"Error during processing: {e}")


def main():
    model = load_model()
    if model is None:
        st.error("Failed to load model.")
        return

    st.sidebar.title("Object Detection")
    type_choice = st.sidebar.selectbox("Select type", ["Image", "Video"])
    file = st.sidebar.file_uploader(
        "Choose a file...",
        type=["jpg", "png", "jpeg"] if type_choice == "Image" else ["mp4", "avi", "mov"]
    )

    if file:
        os.makedirs(RESULT_PATH, exist_ok=True) 

        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}", dir=RESULT_PATH) as temp_file:
            temp_file.write(file.getvalue())
            temp_path = temp_file.name

        result_path = os.path.join(RESULT_PATH, f"result_{file.name}")
        display_and_process_file(model, type_choice, temp_path, result_path)

        # Remove temporary file after processing
        os.remove(temp_path) 
    else:
        st.sidebar.markdown(f"You can download demo files [here]({DEMO_FILES_URL}).")


if __name__ == "__main__":
    main()
