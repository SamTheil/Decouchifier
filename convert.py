import tensorflow as tf

# Load your TensorFlow model
model = tf.keras.models.load_model('ssd_mobilenet_v2_320x320_coco17_tpu-8\saved_model')

# Create a TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open('your_model.tflite', 'wb') as f:
    f.write(tflite_model)