import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

EPOCHS = 15
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
    # print(labels)
    # print(images)

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )
    #Split 1 (80% vs 20%)
    x_train, x_validate_test, y_train, y_validate_test = train_test_split(np.array(images), np.array(labels), test_size=0.2, random_state = 1)
    #Split 2 (50% vs 50%)
    x_test, x_validate, y_test, y_validate = train_test_split(x_validate_test, y_validate_test, test_size=0.50, random_state = 3)
    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_validate, y_validate))

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")
    
    #Plot training loss and training accuracy
    loss_train = history.history['loss']
    loss_val   = history.history['val_loss']
    acc_train  = history.history['accuracy']
    acc_val    = history.history['val_accuracy']
    epochs     = range(1, EPOCHS + 1)

    plot_train_val_history(epochs, loss_train, loss_val,
    "Loss")
    plot_train_val_history(epochs, acc_train, acc_val,
    "Accuracy")


def load_data(data_dir):
    images = []
    labels = []

    # Path to data folder
    data_path = os.path.join(data_dir)

    # Loop through the subdirectories
    for category in range(NUM_CATEGORIES):
        sub_folder = os.path.join(data_path, str(category))

        # Loop through each image in the subdirectory
        for image_name in os.listdir(sub_folder):
            # Path to the image file
            image_path = os.path.join(sub_folder, image_name)

            # Load the image
            image = cv2.imread(image_path)

            # Resize the image to the specified dimensions
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

            # Append the image and label to the lists
            images.append(image)
            labels.append(category)

    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    model = tf.keras.models.Sequential([

        # Convolutional layers and Max-pooling layers
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), 
        tf.keras.layers.Dropout(0.25),        

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25), 
       
        # Flatten units
        tf.keras.layers.Flatten(),        
        # Hidden Layers
        tf.keras.layers.Dense(256, activation='relu'),                
        # Dropout
        tf.keras.layers.Dropout(0.5),        
              
        # Output layer with output units for all digits
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')         
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def plot_train_val_history(x, y_train, y_val, type_txt):
  plt.figure(figsize = (10,7))
  plt.plot(x, y_train, 'g', label='Training'+type_txt)
  plt.title('Training and validation '+type_txt)
  plt.xlabel('Epochs')
  plt.ylabel(type_txt)
  plt.legend()
  plt.savefig('Figure '+type_txt)
  plt.show()
def plot_train_val_history(x, y_train, y_val, type_txt):
  plt.figure(figsize = (10,7))
  plt.plot(x, y_train, 'g', label='Training'+type_txt)
  plt.plot(x, y_val, 'b', label='Validation'+type_txt)
  plt.title('Training and Validation'+type_txt)
  plt.xlabel('Epochs')
  plt.ylabel(type_txt)
  plt.legend()
  plt.savefig('Figure '+type_txt)
  plt.show()
if __name__ == "__main__":
    main()
