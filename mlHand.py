import sys
import io
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Force Python to use UTF-8 encoding to avoid the UnicodeEncodeError
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the datasets
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Check if model already exists
model_path = 'handwritten_model.h5'

# if not os.path.exists(model_path):
#     # Define the model architecture
#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.Input(shape=(28, 28)))  # Input layer (with Input layer instead of Flatten input_shape)
#     model.add(tf.keras.layers.Flatten())  # Flatten layer to prepare for Dense layers
#     model.add(tf.keras.layers.Dense(128, activation='relu'))  # Hidden layer with 128 neurons and ReLU activation
#     model.add(tf.keras.layers.Dense(128, activation='relu'))  # Another hidden layer
#     model.add(tf.keras.layers.Dense(10, activation='softmax'))  # Output layer with 10 neurons (for 10 digits)

#     # Compile the model
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#     # Train the model
#     model.fit(x_train, y_train, epochs=3)

#     # Save the trained model in H5 format
#     model.save(model_path)
#     print(f"Model saved as {model_path}")
# else:
#     # Load the existing model
#     print(f"Loading model from {model_path}")
    
model = tf.keras.models.load_model(model_path)

# # Evaluate the model on test data
# loss, accuracy = model.evaluate(x_test, y_test)
# print(f"Loss: {loss}")
# print(f"Accuracy: {accuracy}")

image_num=1
while os.path.isfile(f"img/d{image_num}.png"):
    try:
        img = cv2.imread(f"img/d{image_num}.png")[:,:,0]
        img=np.invert(np.array([img]))
        prediction=model.predict(img)
        print(f"This digit is may be a {np.argmax(prediction)}")
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error")
    finally:
        image_num+=1
try:
    img = cv2.imread(f"img/digit{image_num}.png")[:,:,0]
    img=np.invert(np.array([img]))
    prediction=model.predict(img)
    print(f"This digit is may be a {np.argmax(prediction)}")
    plt.imshow(img[0],cmap=plt.cm.binary)
    plt.show()
except:
    print("Error")