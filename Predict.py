import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image

# Load the saved model (no retraining)
model = load_model('digit_detector_model.keras')

# Preprocess an image for prediction
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img) / 255.0  # Normalize
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for CNN
    return img_array, img

# Predict digit from a custom image
def predict_digit(image_path):
    img_array, img = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    # Show the image and prediction
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted Label: {predicted_label}")
    plt.axis('off')
    plt.show()

    print(f"Predicted Digit: {predicted_label}")

# Test with images
image_paths = [
    'C:\\Users\\asher\\PycharmProjects\\MNIST\\customeimage\\4.png',
    'C:\\Users\\asher\\PycharmProjects\\MNIST\\customeimage\\5.png',
    'C:\\Users\\asher\\PycharmProjects\\MNIST\\customeimage\\7.png',
    'C:\\Users\\asher\\PycharmProjects\\MNIST\\customeimage\\0.png'
   # 'C:\\Users\\asher\\PycharmProjects\\MNIST\\customeimage\\2.png'
]

for image_path in image_paths:
    predict_digit(image_path)
