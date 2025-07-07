import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Colab에서 다운로드한 모델 경로로 모델 로드
model = tf.keras.models.load_model('./mobilenet_model.keras')

def predict(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions)
    return predicted_class[0], confidence

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    predicted_class, confidence = predict(image_bytes)
    return {"predicted_class": int(predicted_class), "confidence": float(confidence)}
