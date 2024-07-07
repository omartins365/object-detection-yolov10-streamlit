# YOLOv10-Streamlit

This project integrates the YOLOv10 model for object detection into a Streamlit application. The app allows users to upload images or videos and receive object detection results. It provides an interactive interface for users to interact with the model and visualize the results directly in their browser.

### Files and Directories

- **README.md**: Provides an overview and instructions for the project.
- **app**: Contains the main application code.
  - **app.py**: The main Streamlit app that allows users to upload images or videos for object detection.
  - **config.py**: Configuration settings for the app, including paths and URLs.
  - **utils.py**: Utility functions used in the app, such as downloading models and processing images.
- **.devcontainer**: Contains configuration files for the development container.
  - **devcontainer.json**: Configuration file for setting up the development container.
  - **Dockerfile**: Dockerfile for creating the development container image.
- **requirements.txt**: Lists the Python dependencies required for the project.


## Installation

To run this project locally, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/yolov10-streamlit.git
cd yolov10-streamlit
```

### 2. Set Up the Development Container
This project uses a development container for consistent development environments. Ensure you have Docker and Visual Studio Code with the Remote - Containers extension installed.
1. Open the project in Visual Studio Code.
2. Press F1, type Remote-Containers: Open Folder in Container..., and select the project folder.

### 3. Install Dependencies
If not using the development container, you can install the dependencies manually:
```bash
pip install -r requirements.txt
git clone https://github.com/THU-MIG/yolov10.git app/yolov10
pip install -r app/yolov10/requirements.txt
pip install -e app/yolov10
```

### 4. Run the Application
```bash
streamlit run app/app.py
```
Open your browser and navigate to http://localhost:8501 to access the application.

## Usage
1. Choose the type (Image or Video) from the sidebar.
2. Upload an image or video file using the file uploader.
3. View the uploaded file and its processed result in the app interface.

Results are saved in the app/result folder.