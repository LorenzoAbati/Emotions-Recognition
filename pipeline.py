import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Disable eager execution to potentially use GPU on Mac M1
tf.config.run_functions_eagerly(False)


# 1. Data Preprocessing
base_dir = "./train/"  # Assuming the script runs at the parent directory containing the emotion folders
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
labels = []
images = []

for i, emotion in enumerate(emotions):
    emotion_dir = os.path.join(base_dir, emotion)
    for image_path in os.listdir(emotion_dir):
        full_image_path = os.path.join(emotion_dir, image_path)
        image = load_img(full_image_path, target_size=(48, 48))
        image = img_to_array(image)
        images.append(image)
        labels.append(i)

images = np.array(images) / 255.0
labels = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

from tensorflow.keras.callbacks import TensorBoard
import datetime

# Set up TensorBoard logging directory
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# 2. Model Building
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax')  # 7 for the number of emotions
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. Training
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])


# 4. Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)

# 5. Load and preprocess test images from ./test directory
test_base_dir = "./test/"
test_labels = []
test_images = []

for i, emotion in enumerate(emotions):
    emotion_dir = os.path.join(test_base_dir, emotion)
    for image_path in os.listdir(emotion_dir):
        full_image_path = os.path.join(emotion_dir, image_path)
        image = load_img(full_image_path, target_size=(48, 48))
        image = img_to_array(image)
        test_images.append(image)
        test_labels.append(i)

test_images = np.array(test_images) / 255.0
test_labels = np.array(test_labels)

# 6. Evaluate model using the new test images
test_loss_external, test_acc_external = model.evaluate(test_images, test_labels)
print("Test accuracy on external test set:", test_acc_external)


# Saving the model
model.save('emotion_recognition_model.keras')


