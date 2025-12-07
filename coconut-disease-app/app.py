import os
import uuid
import logging

from flask import Flask, render_template, request
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- Config / Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("coconut-app")

# --- Flask app ---
app = Flask(__name__)

# --- Paths ---
UPLOAD_FOLDER = "static/uploads/"
CAM_FOLDER = "static/cam_results/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CAM_FOLDER, exist_ok=True)

# --- Settings ---
IMG_SIZE = 224
CLASS_NAMES = ["Bud_Rot", "Coconut_Caterpillar", "Leaf_Rot", "Stem_Bleeding", "Yellow_Leaf_Disease"]
MODEL_PATH = "model/MobileNetV2_Coconut_Model.h5"

# --- Load model ---
logger.info(f"Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)
logger.info("Model loaded.")

# --- Find last conv layer (MobileNetV2 typically has 'Conv_1' but we search robustly) ---
def find_last_conv_layer(m):
    for layer in reversed(m.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            return layer.name
    for layer in reversed(m.layers):
        if "conv" in layer.name.lower():
            return layer.name
    raise ValueError("No convolutional layer found in model")

last_conv_layer_name = find_last_conv_layer(model)
logger.info(f"Using last conv layer: {last_conv_layer_name}")

# --- Grad-CAM implementation ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """
    img_array: shape (1, H, W, C) float32 scaled 0..1
    returns: heatmap (2D numpy 0..1) and pred_index (python int)
    """
    # 1) get predictions via numpy and choose predicted class index safely
    preds = model.predict(img_array)  # shape (1, num_classes)
    logger.info(f"Predictions array shape: {preds.shape}")
    pred_index = int(np.argmax(preds[0]))  # ALWAYS a python int
    logger.info(f"Predicted class index (int): {pred_index}, prob: {preds[0][pred_index]:.4f}")

    # 2) build a grad model to fetch conv outputs and predictions (tf tensors)
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 3) compute gradients for the predicted class w.r.t last conv layer output
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        # use pred_index (python int) to select the class score
        predictions_tensor = predictions[0] if isinstance(predictions, list) else predictions
        class_channel = predictions_tensor[:, pred_index]


    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        # Shouldn't happen normally; safe fallback
        raise RuntimeError("Gradients are None: check model and last_conv_layer_name.")

    # 4) global average pooling on the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape (channels,)

    # 5) weight conv_outputs with pooled grads
    conv_outputs = conv_outputs[0]  # shape (h, w, channels)
    heatmap = tf.tensordot(conv_outputs, pooled_grads, axes=([2], [0]))  # shape (h, w)

    # 6) relu and normalize to 0..1
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0 or tf.math.is_nan(max_val):
        heatmap = tf.zeros_like(heatmap)
    else:
        heatmap = heatmap / (max_val + 1e-10)

    heatmap_np = heatmap.numpy()
    return heatmap_np, pred_index


def overlay_heatmap(heatmap, img_path, intensity=0.5):
    """
    heatmap: 2D numpy array 0..1
    returns: original (RGB numpy), heatmap_color (BGR), superimposed (BGR)
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read uploaded image at {img_path}")

    img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))

    # resize heatmap to image size and convert to uint8
    heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img_bgr, 1 - intensity, heatmap_color, intensity, 0)

    original_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return original_rgb, heatmap_color, superimposed


# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]
        if file.filename == "":
            return "No image selected", 400

        # save uploaded file
        filename = str(uuid.uuid4()) + ".jpg"
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(img_path)
        logger.info(f"Saved uploaded image to: {img_path}")

        # preprocess for model
        img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, C)

        # generate heatmap and pred index
        heatmap, pred_index = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        pred_index = int(pred_index)  # ensure python int

        # overlay and save results
        original_rgb, heatmap_bgr, cam_bgr = overlay_heatmap(heatmap, img_path, intensity=0.5)
        heatmap_file = filename.replace(".jpg", "_heatmap.jpg")
        cam_file = filename.replace(".jpg", "_cam.jpg")

        heatmap_path = os.path.join(CAM_FOLDER, heatmap_file)
        cam_path = os.path.join(CAM_FOLDER, cam_file)
        cv2.imwrite(heatmap_path, heatmap_bgr)
        cv2.imwrite(cam_path, cam_bgr)

        predicted_label = CLASS_NAMES[pred_index]

        # pass relative paths to template (Flask serves from static/)
        return render_template(
            "result.html",
            original_image=img_path,
            heatmap_image=heatmap_path,
            cam_image=cam_path,
            predicted=predicted_label
        )

    except Exception as e:
        logger.exception("Error during prediction")
        return f"Error during prediction: {str(e)}", 500


if __name__ == "__main__":
    app.run(debug=True)
