import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import os
import random

# Load the trained model
model = tf.keras.models.load_model('emotion_recognition_model.h5')

emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def predict_emotion(img_path, model):
    """
    Load and preprocess the image from the provided path and predict its emotion using the model.

    Args:
    - img_path (str): Path to the image file
    - model (tf.keras.Model): Trained emotion recognition model

    Returns:
    - str: Predicted emotion
    """
    # Load and preprocess the image
    image = load_img(img_path, target_size=(48, 48))
    image_array = img_to_array(image)
    image_batch = np.expand_dims(image_array, axis=0)  # Increase dimension for batch size
    normalized_image = image_batch / 255.0  # Normalize

    # Predict emotion
    predictions = model.predict(normalized_image)
    predicted_emotion = emotions[np.argmax(predictions)]

    return predicted_emotion, image_array


# Continuous prompt for image directories
while True:
    emotion_dir_input = input("Enter the directory containing emotion images (or 'exit' to quit): ")

    if emotion_dir_input.lower() == 'exit':
        break

    full_path = os.path.join('./test', emotion_dir_input)
    if not os.path.exists(full_path):
        print(f"The directory '{full_path}' doesn't exist.")
        continue

    random_image = random.choice(os.listdir(full_path))
    img_path = os.path.join(full_path, random_image)

    predicted_emotion, image_array = predict_emotion(img_path, model)

    # Show the image
    plt.imshow(image_array.astype('uint8'))
    plt.title(f"Predicted emotion: {predicted_emotion}")
    plt.axis('off')  # hide the axis
    plt.show()
