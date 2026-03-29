from flask import Flask, render_template, request, send_file, session
import os
import numpy as np
from werkzeug.utils import secure_filename
from keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime

from predictions import (
    predict,
    model_elbow_frac,
    model_hand_frac,
    model_shoulder_frac
)
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


def get_real_confidence(model, img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)

    preds = model.predict(img_arr)
    if isinstance(preds, list):
        preds = preds[0]

    class_idx = np.argmax(preds[0])
    confidence = float(preds[0][class_idx]) * 100

    status = "fractured" if class_idx == 0 else "normal"
    return status, round(confidence, 2)


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

        # BODY PART DETECTION
        region = predict(image_path, model="Parts")

        # FRACTURE MODEL
        used_model = {
            "Elbow": model_elbow_frac,
            "Hand": model_hand_frac,
            "Shoulder": model_shoulder_frac
        }[region]

        status, confidence = get_real_confidence(used_model, image_path)

        # MEDICAL SAFETY
        if status == "normal" and confidence < 65:
            status = "suspected fracture"

        roi_url = None
        gradcam_url = None

        # Generate explainability only if fracture
        if status in ["fractured", "suspected fracture"]:

            roi_path = os.path.join(RESULT_FOLDER, f"roi_{filename}")
            generate_roi(image_path, roi_path)
            roi_url = "/static/results/" + os.path.basename(roi_path)

            grad_layer = get_last_conv_layer(used_model)
            gradcam_path = os.path.join(RESULT_FOLDER, f"gradcam_{filename}")

            if grad_layer:
                generate_gradcam(
                    model=used_model,
                    img_path=image_path,
                    layer_name=grad_layer,
                    save_path=gradcam_path
                )
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

        # STORE SESSION
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
# PDF GENERATION ROUTE
# --------------------------------------------------
@app.route("/download-medical-report")
def download_medical_report():

    if "report" not in session:
        return "Analyze image first before downloading report."

    report = session["report"]

    region = report["region"]
    status = report["status"]
    confidence = report["confidence"]
    severity = report["severity"]
    displacement = report["displacement"]
    surgery = report["surgery"]
    actions = report["actions"]
    roi_url = report["roi"]
    gradcam_url = report["gradcam"]

    pdf_path = os.path.join(BASE_DIR, "static", "fracture_report.pdf")

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    def draw_page_border(canvas, doc):
        canvas.saveState()
        width, height = A4
        canvas.setLineWidth(2)
        canvas.rect(20, 20, width-40, height-40)
        canvas.restoreState()

    # HEADER
    story.append(Paragraph("FractureAI Diagnostic Center", styles['Title']))
    story.append(Paragraph("AI-Assisted Radiology Report", styles['Heading2']))
    story.append(Spacer(1, 15))

    # PATIENT INFO
    story.append(Paragraph("<b>Patient Information</b>", styles['Heading3']))
    patient_table = Table([
        ["Patient ID:", "AUTO-GENERATED"],
        ["Scan Date:", datetime.now().strftime("%d %b %Y")],
        ["Exam Type:", "X-Ray Imaging"],
        ["Analyzed by:", "FractureAI AI Model"]
    ], colWidths=[150, 300])

    patient_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BOX', (0,0), (-1,-1), 2, colors.black),
        ('BACKGROUND', (0,0), (0,-1), colors.lightgrey)
    ]))

    story.append(patient_table)
    story.append(Spacer(1, 20))

    # CLINICAL FINDINGS
    story.append(Paragraph("<b>Clinical Findings</b>", styles['Heading3']))
    findings_table = Table([
        ["Anatomical Region", region],
        ["Fracture Status", status],
        ["Confidence Score", f"{confidence}%"],
        ["Severity", severity],
        ["Displacement", f"{displacement}°"],
        ["Surgery Recommendation", surgery]
    ], colWidths=[200, 250])

    findings_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BOX', (0,0), (-1,-1), 2, colors.black),
        ('BACKGROUND', (0,0), (0,-1), colors.lightgrey)
    ]))

    story.append(findings_table)
    story.append(Spacer(1, 25))

    # ONLY ADD RADIOLOGICAL SECTION IF FRACTURE
    if status in ["fractured", "suspected fracture"] and roi_url and gradcam_url:

        story.append(Paragraph("<b>Radiological Evidence</b>", styles['Heading3']))
        story.append(Spacer(1, 12))

        roi_path = os.path.join(BASE_DIR, roi_url.strip("/"))
        gradcam_path = os.path.join(BASE_DIR, gradcam_url.strip("/"))

        image_table_data = [
            [
                Paragraph("<b>Region of Interest</b>", styles['Normal']),
                Paragraph("<b>Grad-CAM Heatmap</b>", styles['Normal'])
            ],
            [
                Image(roi_path, width=3.2*inch, height=3.2*inch),
                Image(gradcam_path, width=3.2*inch, height=3.2*inch)
            ]
        ]

        img_table = Table(image_table_data, colWidths=[260, 260], rowHeights=[35, 240])

        img_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('BOX', (0,0), (-1,-1), 2, colors.black),
            ('ALIGN', (0,0), (-1,0), 'CENTER'),
            ('ALIGN', (0,1), (-1,1), 'CENTER'),
            ('VALIGN', (0,1), (-1,1), 'MIDDLE'),
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey)
        ]))

        story.append(img_table)
        story.append(Spacer(1, 30))

    # RECOMMENDATIONS
    story.append(Paragraph("<b>Clinical Recommendation</b>", styles['Heading3']))
    for action in actions:
        story.append(Paragraph(f"• {action}", styles['Normal']))

    story.append(Spacer(1, 20))

    story.append(Paragraph("<b>Medical Disclaimer</b>", styles['Heading3']))
    story.append(Paragraph(
        "This AI-generated report must be verified by a licensed medical professional.",
        styles['Normal']
    ))

    doc.build(story, onFirstPage=draw_page_border, onLaterPages=draw_page_border)

    return send_file(pdf_path, as_attachment=True)


# --------------------------------------------------
# RUN SERVER
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
