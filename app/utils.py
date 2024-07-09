import os

import streamlit as st
import wget
from config import MODEL_URL
from ultralytics import YOLOv10


def download_model(model_path: str) -> bool:
    """ Download the YOLOv10 model if it doesn't exist. """
    if not os.path.exists(model_path):  # Check if model_path does not exist  
        os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Create directory if it doesn't exist
        try:
            with st.spinner("Downloading YOLOv10 model..."):
                wget.download(MODEL_URL, model_path)  # Download the model from MODEL_URL to model_path
                st.success("Downloaded YOLOv10 model successfully.")
                return True
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            return False
    return True


def load_model(model_path: str) -> YOLOv10:
    """Load the YOLOv10 model."""
    return YOLOv10(model_path)
