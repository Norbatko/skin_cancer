import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Function to load and preprocess the image
def preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)  # Load image and resize to target size
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for batch size
    img_array /= 255.0  # Rescale the image (just like you did during training)
    return img_array

# Main function
def main():
    tf.config.set_visible_devices([], 'GPU')
    # Load the trained model
    model = tf.keras.models.load_model('cnn_model.h5')

    # Path to your image
    img_path = '../data/benign/ISIC_0000130.jpg'  # Replace with your image path

    # Preprocess the image
    img_array = preprocess_image(img_path)

    # Make a prediction
    prediction = model.predict(img_array)

    # Interpret the result (assuming binary classification: 0 = benign, 1 = malign)
    if prediction[0] > 0.5:
        print("Predicted: Malignant")
    else:
        print("Predicted: Benign")

    print(f"Prediction score: {prediction[0]}")

    # Show the image
    img = image.load_img(img_path)

if __name__ == "__main__":
    main()
