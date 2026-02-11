import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

model_elbow_frac = tf.keras.models.load_model("weights/EfficientNetB1_Elbow.h5")
model_hand_frac = tf.keras.models.load_model("weights/EfficientNetB1_Hand.h5")
model_shoulder_frac = tf.keras.models.load_model("weights/EfficientNetB1_Shoulder.h5")
model_parts = tf.keras.models.load_model("weights/EfficientNetB1_BodyParts.h5")

categories_parts = ["Elbow", "Hand", "Shoulder"]
categories_fracture = ["fractured", "normal"]

def predict(img_path, model="Parts"):
    size = 224

    img = image.load_img(img_path, target_size=(size, size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    if model == "Parts":
        x = effnet_preprocess(x)
        preds = model_parts.predict(x)
        return categories_parts[np.argmax(preds[0])]

    else:
        x = effnet_preprocess(x)

        if model == "Elbow":
            preds = model_elbow_frac.predict(x)
        elif model == "Hand":
            preds = model_hand_frac.predict(x)
        elif model == "Shoulder":
            preds = model_shoulder_frac.predict(x)

        return categories_fracture[np.argmax(preds[0])]
