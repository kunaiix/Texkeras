import keras_ocr
import matplotlib.pyplot as plt
import cv2
import numpy as np
from spellchecker import SpellChecker
pipeline = keras_ocr.pipeline.Pipeline()

image_path = 'testimg1.jpeg'

image_pros = cv2.imread(image_path)
gray_image = cv2.cvtColor(image_pros, cv2.COLOR_BGR2GRAY)
enhanced_image = cv2.addWeighted(gray_image, 1, np.zeros(gray_image.shape, gray_image.dtype), 0, 0)
final_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
image = keras_ocr.tools.read(final_image)

prediction_groups = pipeline.recognize([image])

spell = SpellChecker()

corrected_texts = []
for prediction in prediction_groups[0]:
    text, box = prediction
    print(f"Detected text: {text}")

    corrected_words = []
    for word in text.split():
        corrected_word = spell.correction(word)
        corrected_words.append(corrected_word)

    corrected_text = ' '.join(corrected_words)
    corrected_texts.append(corrected_text)
    print(f"Corrected text: {corrected_text}")


for prediction in prediction_groups[0]:
    text, box = prediction
    print(f"Detected text: {text}")

for corrected_text in corrected_texts:
    print(f"Final corrections: {corrected_text}")

keras_ocr.tools.drawAnnotations(image=image, predictions=prediction_groups[0])
plt.imshow(image)
plt.show()


