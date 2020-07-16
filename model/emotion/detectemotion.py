import os
import sys
import cv2
import numpy as np
from keras.models import load_model
from .utils.inference import draw_text
from .utils.datasets import get_labels
from .utils.inference import load_image
from .utils.inference import detect_faces
from .utils.inference import apply_offsets
from .utils.inference import draw_bounding_box
from .utils.preprocessor import preprocess_input
from .utils.inference import load_detection_model

# parameters for loading data and images
emotion_model_path = 'model/emotion/emotion.hdf5'
# loading models
emotion_classifier = load_model(emotion_model_path, compile=False)

class Emotional():
    def __init__(self):
        self.emotion_model_path = emotion_model_path

    """ Detect the emotion of the cropped images
        Emotions are 
        1. Happy
        2. Neutral
        3. Surprise
        4. Angry
        5. Fear
        6. Sad
        7. Disgust

    """
    def emotionalDet(self,imagepath,faceBox):
        
        emotion_model_path = self.emotion_model_path
        emotion_labels = get_labels('fer2013')
        gender_labels = get_labels('imdb')
        font = cv2.FONT_HERSHEY_SIMPLEX
        # hyper-parameters for bounding boxes shape
        emotion_offsets = (20, 40)
        emotion_offsets = (0, 0)
        
        # getting inputfor inference
        emotion_target_size = emotion_classifier.input_shape[1:3]
        # loading images
        rgb_image = load_image(imagepath, grayscale=False)
        gray_image = load_image(imagepath, grayscale=True)
        gray_image = np.squeeze(gray_image)
        gray_image = gray_image.astype('uint8')
        filename1 = os.path.basename(imagepath)
        faceEmotionList = []
        count = 0
        # for face_coordinates in faces:
        if faceBox:
            count += 1
            x1, x2, y1, y2 = apply_offsets(faceBox, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                print("Something went wrong")
            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
            emotion_text = emotion_labels[emotion_label_arg]
            faceEmotionList.append(emotion_text)
        return faceEmotionList