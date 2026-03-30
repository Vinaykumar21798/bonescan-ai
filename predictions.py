import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Lazy loading
model_elbow = None
model_hand = None
model_shoulder = None
model_parts = None

categories_parts = ["Elbow", "Hand", "Shoulder"]
categories_fracture = ["fractured", "normal"]


def load_models():
    global model_elbow, model_hand, model_shoulder, model_parts

    if model_parts is None:
        model_parts = tf.keras.models.load_model("weights/EfficientNetB1_BodyParts.h5")

    if model_elbow is None:
        model_elbow = tf.keras.models.load_model("weights/EfficientNetB1_Elbow.h5")

    if model_hand is None:
        model_hand = tf.keras.models.load_model("weights/EfficientNetB1_Hand.h5")

    if model_shoulder is None:
        model_shoulder = tf.keras.models.load_model("weights/EfficientNetB1_Shoulder.h5")


def predict(img_path):
    load_models()

    size = 224
    img = image.load_img(img_path, target_size=(size, size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Step 1: detect body part
    part_pred = model_parts.predict(x)
    region = categories_parts[np.argmax(part_pred[0])]

    # Step 2: fracture detection
    if region == "Elbow":
        preds = model_elbow.predict(x)
    elif region == "Hand":
        preds = model_hand.predict(x)
    else:
        preds = model_shoulder.predict(x)

    class_idx = np.argmax(preds[0])
    confidence = float(preds[0][class_idx]) * 100
    status = categories_fracture[class_idx]

    return region, status, round(confidence, 2)