from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf
import numpy as np
import cv2


def generate_gradcam(model, img_path, layer_name="top_conv", save_path="gradcam.jpg"):

    original = cv2.imread(img_path)
    original = cv2.resize(original, (224, 224))

    img = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img.astype(np.float32))

    # IMPORTANT FIX
    model_output = model.output
    if isinstance(model_output, list):
        model_output = model_output[0]

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model_output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)

        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original, 0.65, heatmap, 0.35, 0)

    cv2.imwrite(save_path, overlay)
    return save_path


# --------------------------------------------------
# ROI EXTRACTION (UNCHANGED, STABLE)
# --------------------------------------------------
def generate_roi(img_path, save_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        roi = img[y:y + h, x:x + w]
    else:
        roi = img

    cv2.imwrite(save_path, roi)
    return save_path
