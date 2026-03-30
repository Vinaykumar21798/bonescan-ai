from flask import Flask, render_template, request, send_file, session
import os
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime

from predictions import predict  # ✅ ONLY this import
from visual_explainability import generate_gradcam, generate_roi

# --------------------------------------------------
# APP INITIALIZATION
# --------------------------------------------------
app = Flask(__name__)
app.secret_key = "fracture_secret_key"

BASE_DIR = app.root_path

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# --------------------------------------------------
# UTILITIES
# --------------------------------------------------
def get_last_conv_layer(model):
    return "top_conv"


# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/features")
def features():
    return render_template("features.html")


@app.route("/about-model")
def about_model():
    return render_template("about-model.html")


@app.route("/faq")
def faq():
    return render_template("faq.html")


# --------------------------------------------------
# PREDICTION ROUTE
# --------------------------------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict_image():

    if request.method == "POST":

        if "xray_image" not in request.files:
            return "No image uploaded", 400

        file = request.files["xray_image"]
        if file.filename == "":
            return "Empty filename", 400

        filename = secure_filename(file.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(image_path)

        # ✅ Single unified prediction
        region, status, confidence = predict(image_path)

        roi_url = None
        gradcam_url = None

        if status in ["fractured", "suspected fracture"]:

            roi_path = os.path.join(RESULT_FOLDER, f"roi_{filename}")
            generate_roi(image_path, roi_path)
            roi_url = "/static/results/" + os.path.basename(roi_path)

            gradcam_path = os.path.join(RESULT_FOLDER, f"gradcam_{filename}")
            generate_gradcam(image_path, gradcam_path)
            gradcam_url = "/static/results/" + os.path.basename(gradcam_path)

        # CLINICAL LOGIC
        if status in ["fractured", "suspected fracture"]:
            severity = "High" if confidence > 70 else "Moderate"
            surgery = "Likely Yes" if severity == "High" else "Possibly"
            displacement = round(confidence * 0.6, 1)
            actions = [
                "Emergency medical care required",
                "Consult orthopedic specialist",
                "Avoid movement and weight bearing",
                "Follow-up X-ray after stabilization"
            ]
        else:
            severity = "None"
            surgery = "No"
            displacement = 0.0
            actions = [
                "No fracture detected",
                "Normal activity can be resumed",
                "Follow-up only if pain persists"
            ]

        session["report"] = {
            "region": region,
            "status": status,
            "confidence": confidence,
            "severity": severity,
            "displacement": displacement,
            "surgery": surgery,
            "actions": actions,
            "roi": roi_url,
            "gradcam": gradcam_url
        }

        return render_template(
            "result.html",
            region=region,
            status=status,
            confidence=confidence,
            displacement=displacement,
            severity=severity,
            surgery=surgery,
            actions=actions,
            roi=roi_url,
            gradcam=gradcam_url
        )

    return render_template("predict.html")


# --------------------------------------------------
# PDF ROUTE
# --------------------------------------------------
@app.route("/download-medical-report")
def download_medical_report():

    if "report" not in session:
        return "Analyze image first."

    report = session["report"]

    pdf_path = os.path.join(BASE_DIR, "static", "fracture_report.pdf")

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("FractureAI Diagnostic Center", styles['Title']))
    story.append(Spacer(1, 20))

    story.append(Paragraph(f"Region: {report['region']}", styles['Normal']))
    story.append(Paragraph(f"Status: {report['status']}", styles['Normal']))
    story.append(Paragraph(f"Confidence: {report['confidence']}%", styles['Normal']))

    doc.build(story)

    return send_file(pdf_path, as_attachment=True)


# --------------------------------------------------
# RUN SERVER (IMPORTANT 🔥)
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)