import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from tensorflow.keras.models import load_model
from PIL import Image

# Load model once when the server starts
# Assumes cifar10_cnn_model.h5 is in the SAME folder as manage.py
model = load_model("cifar10_cnn_model.h5")

# CIFAR-10 class labels
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

@csrf_exempt
def predict(request):
    if request.method == "POST":
        if "file" not in request.FILES:
            return JsonResponse({"error": "No file uploaded"}, status=400)

        try:
            # Read the uploaded image
            file = request.FILES["file"]
            image = Image.open(file)

            # Convert to RGB
            image = image.convert("RGB")

            # Resize to CIFAR-10 input size
            image = image.resize((32, 32))

            # Convert to numpy and normalize
            img_array = np.array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # (1, 32, 32, 3)

            # Predict
            predictions = model.predict(img_array)
            class_index = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))

            return JsonResponse({
                "filename": file.name,
                "predicted_class": CLASS_NAMES[class_index],
                "confidence": round(confidence, 4)
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)
