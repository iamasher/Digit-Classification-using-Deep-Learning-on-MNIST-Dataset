import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(x_train)

# Define the hypermodel for Keras Tuner
class CNNHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Conv2D(
            filters=hp.Int('filters', min_value=32, max_value=128, step=32),
            kernel_size=3,
            activation='relu',
            input_shape=(28, 28, 1)
        ))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(
            units=hp.Int('units', min_value=32, max_value=128, step=32),
            activation='relu'
        ))
        model.add(Dense(10, activation='softmax'))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

# Initialize the Keras Tuner
tuner = RandomSearch(
    CNNHyperModel(),
    objective='val_accuracy',
    max_trials=2,
    executions_per_trial=1,
    directory='hyperparameter_tuning',
    project_name='mnist_cnn'
)

# Perform hyperparameter tuning
tuner.search(datagen.flow(x_train, y_train, batch_size=128),
             epochs=10,
             validation_data=(x_test, y_test))

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the best model on the test set
test_loss, test_accuracy = best_model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy:.4f}')

# Generate predictions and compute the confusion matrix
y_pred = best_model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print(classification_report(y_true, y_pred_classes))

# Save the best model
best_model.save('mnist_cnn_best_model.keras')
