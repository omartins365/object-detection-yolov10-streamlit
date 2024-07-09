from config import (
    BATCH_SIZE,
    EPOCHS,
    IMG_SIZE,
    MODEL_PATH,
    TRAINED_MODEL_PATH,
    YAML_PATH,
)
from ultralytics import YOLOv10


def train():
    """Train YOLOv10 model."""
    model = YOLOv10(MODEL_PATH)
    model.train(data=YAML_PATH,
                epochs=EPOCHS,
                batch=BATCH_SIZE,
                imgsz=IMG_SIZE)
    
def val():
    """Validate YOLOv10 model."""
    model = YOLOv10(TRAINED_MODEL_PATH)
    model.val(data=YAML_PATH,
              imgsz=IMG_SIZE,
              split='test')
    
if "__main__" == __name__:
    train()
    val()