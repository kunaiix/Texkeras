import keras_ocr
import matplotlib.pyplot as plt
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

pipeline = keras_ocr.pipeline.Pipeline()

image_path = 'testimg2.png'
image = keras_ocr.tools.read(image_path)

prediction_groups = pipeline.recognize([image])

for prediction in prediction_groups[0]:
    text, box = prediction
    print(f"Detected text: {text}")

keras_ocr.tools.drawAnnotations(image=image, predictions=prediction_groups[0])
plt.imshow(image)
plt.show()
