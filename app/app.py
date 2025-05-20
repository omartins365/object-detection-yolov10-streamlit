import gradio as gr
import cv2
import os
import tempfile
from PIL import Image
from config import (
    DEMO_FILES_URL,
    IMAGE_WIDTH,
    MODEL_PATH,
    RESULT_PATH,
    TRAINED_MODEL_PATH,
)
from ultralytics import YOLOv10
from utils import download_model, load_model


def process_image(model: YOLOv10, image: Image.Image) -> Image.Image:
    result = model(source=image)[0]
    return Image.fromarray(result.plot())


def process_video(model: YOLOv10, video_path: str) -> str:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video_path = tempfile.mktemp(suffix=".webm")
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'vp80'), fps, (frame_width, frame_height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
    cap.release()
    out.release()
    return output_video_path


def infer(image, video, input_type, model_choice):
    model_path = TRAINED_MODEL_PATH if model_choice == "Custom Trained Model" else MODEL_PATH
    if model_choice == "Default YOLOv10":
        download_model(MODEL_PATH)
    model = load_model(model_path)
    if input_type == "Image" and image is not None:
        return process_image(model, image), None
    elif input_type == "Video" and video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(video.read())
            temp_path = temp_file.name
        result_video_path = process_video(model, temp_path)
        return None, result_video_path
    else:
        return None, None


def gradio_app():
    with gr.Blocks() as demo:
        gr.Markdown("# YOLOv10 Object Detection (Gradio)")
        with gr.Row():
            input_type = gr.Radio(["Image", "Video"], value="Image", label="Input Type")
            model_choice = gr.Dropdown(["Default YOLOv10", "Custom Trained Model"], value="Default YOLOv10", label="Model")
        with gr.Row():
            image = gr.Image(type="pil", label="Input Image")
            video = gr.Video(label="Input Video")
        with gr.Row():
            output_image = gr.Image(type="pil", label="Annotated Image")
            output_video = gr.Video(label="Annotated Video")
        def update_inputs(input_type):
            return (
                gr.update(visible=input_type=="Image"),
                gr.update(visible=input_type=="Video"),
                gr.update(visible=input_type=="Image"),
                gr.update(visible=input_type=="Video")
            )
        input_type.change(update_inputs, [input_type], [image, video, output_image, output_video])
        btn = gr.Button("Detect Objects")
        btn.click(
            fn=infer,
            inputs=[image, video, input_type, model_choice],
            outputs=[output_image, output_video],
        )
        gr.Markdown(f"[Download demo files here]({DEMO_FILES_URL})")
    return demo


if __name__ == "__main__":
    gradio_app().launch(server_port=7860, server_name="0.0.0.0")
