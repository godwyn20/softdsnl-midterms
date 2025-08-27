import tensorflow as tf
import numpy as np
from rest_framework.decorators import api_view
from rest_framework.response import Response
from PIL import Image

# Load model once
model = tf.keras.models.load_model("cifar10_cnn_model.h5")
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def preprocess_image(image_file):
    img = Image.open(image_file).resize((32,32))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1,32,32,3)
    return img_array

@api_view(["POST"])
def predict(request):
    if "image" not in request.FILES:
        return Response({"error": "No image provided"}, status=400)

    img_array = preprocess_image(request.FILES["image"])
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    return Response({"prediction": predicted_class})