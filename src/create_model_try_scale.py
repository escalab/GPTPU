import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib
import os
print(tf.__version__)
print("WARNING: This script is only tested on tf==2.2.0.")

def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1, 28, 28)
      yield [data.astype(np.float32)]
# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#train_images = train_images[:1,:,:]
#train_labels = train_labels[1]
#test_images = test_images[:1,:,:]
#test_labels = test_labels[1]

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model architecture
model = keras.Sequential([
  keras.layers.InputLayer(input_shape=(28, 28)),
  keras.layers.Reshape(target_shape=(28, 28, 1)),
  keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation=tf.nn.relu),
  keras.layers.MaxPooling2D(pool_size=(2, 2)),
  keras.layers.Flatten(),
  keras.layers.Dense(1)
])

# Train the digit classification model
model.compile(optimizer='adam',
              loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(
  train_images[:1,:,:],
  train_labels[:1],
  epochs=1
  #validation_data=(test_images, test_labels)
)

# ===== Convert to a TensorFlow Lite model =====

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_models_dir = pathlib.Path("/tmp/mnist_tflite_models/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

tflite_model_file = tflite_models_dir/"mnist_model.tflite"
tflite_model_file.write_bytes(tflite_model)

converter.experimental_new_quantizer = False
converter.optimizations = [tf.lite.Optimize.DEFAULT]

 
converter.representative_dataset = representative_dataset
converter.inference_input_type = tf.uint8  # or tf.uint8
converter.inference_output_type = tf.uint8  # or tf.uint8

tflite_model = converter.convert()
print("convert done.")
tflite_model_file = tflite_models_dir/"mnist_model_quant.tflite"
tflite_model_file.write_bytes(tflite_model)
# ===== Run the Tensorflow Lite models =====

interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()

test_image = np.expand_dims(test_images[0], axis=0).astype(np.float32)

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

interpreter.set_tensor(input_index, test_image)
interpreter.invoke()
predictions = interpreter.get_tensor(output_index)


# helper function to evaluate the TF Lite model using "test" dataset.
def evaluate_model(interpreter):
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]

  # Run predictions on every image in the "test" dataset.
  prediction_digits = []
  for test_image in test_images:
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, test_image)

    # Run inference.
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    output = interpreter.tensor(output_index)
    digit = np.argmax(output()[0])
    prediction_digits.append(digit)

  # Compare prediction results with ground truth labels to calculate accuracy.
  accurate_count = 0
  for index in range(len(prediction_digits)):
    if prediction_digits[index] == test_labels[index]:
      accurate_count += 1
  accuracy = accurate_count * 1.0 / len(prediction_digits)

  return accuracy


#print(evaluate_model(interpreter))
# NOTE: This quantization mode is an experimental post-training mode,
# it does not have any optimized kernels implementations or
# specialized machine learning hardware accelerators. Therefore,
# it could be slower than the float interpreter.
#print(evaluate_model(interpreter))

os.system("edgetpu_compiler "+str(tflite_model_file)+" -o ./ -s")

