# bonescan-ai
🦴 FractureAI – Hierarchical Bone Fracture Detection System
📌 Overview

FractureAI is a deep learning–based, hierarchical bone fracture detection system developed using the MURA (Musculoskeletal Radiographs) dataset.

The system automatically detects fractures from X-ray images using EfficientNetB1-based convolutional neural networks, integrates Explainable AI (Grad-CAM + ROI), and provides an AI-assisted radiology report through a Flask-based web application.

This project bridges the gap between research and real-world deployment by combining:

Hierarchical classification

Bone-specific fracture detection

Explainable AI

Web deployment

Automated PDF reporting

🚀 Key Features

✅ Hierarchical anatomical region classification

✅ Bone-specific fracture detection models

✅ Transfer learning using EfficientNetB1

✅ Grad-CAM explainability

✅ ROI extraction for visual localization

✅ Confidence-aware prediction logic

✅ Flask web application deployment

✅ AI-generated PDF radiology reports

🏗️ System Architecture

The system follows a two-stage hierarchical approach:

🔹 Stage 1 – Anatomical Region Classification

Input X-ray → EfficientNetB1 →
Output: Elbow | Hand | Shoulder

🔹 Stage 2 – Bone-Specific Fracture Detection

Based on anatomical prediction →
Route to corresponding EfficientNetB1 binary classifier →
Output: Fractured / Normal

🔹 Additional Modules

Confidence-aware safety logic

Grad-CAM heatmap generation

ROI extraction

PDF report generation

📊 Dataset

Dataset: MURA (Musculoskeletal Radiographs)

Regions used:

Elbow

Hand

Shoulder

Labels:

Fractured (Abnormal)

Normal

Custom directory traversal is used for dataset loading and label assignment.

🧠 Model Details

Backbone: EfficientNetB1

Transfer Learning: ImageNet pretrained weights

Input Size: 224 × 224

Training Strategy:

Frozen base layers initially

Fine-tuning on MURA dataset

Separate models are trained for:

Elbow fracture detection

Hand fracture detection

Shoulder fracture detection

Multi-class anatomical region classification

🔎 Explainable AI (XAI)

To enhance transparency and clinical trust:

Grad-CAM highlights regions influencing predictions

ROI extraction identifies key structural areas

These visual explanations are displayed in the web interface and included in the PDF report.

🌐 Deployment

The system is deployed using:

Backend: Flask (app.py)

Frontend: HTML + CSS templates

Inference: Pre-trained EfficientNetB1 models

Output: Prediction + confidence + PDF report

⚠️ Training is performed offline.
Deployment performs inference only.

📂 Project Structure
FractureAI/
│
├── app.py                     # Flask web application
├── predictions.py             # Model loading & inference
├── visual_explainability.py   # Grad-CAM & ROI logic
├── requirements.txt           # Dependencies
│
├── templates/                 # HTML files
│   ├── index.html
│   ├── predict.html
│   ├── result.html
│   ├── about-model.html
│   └── faq.html
│
├── static/
│   └── style.css
│
├── models/                    # Trained .h5/.keras models
│
└── notebooks/
    ├── EfficientNetB1_Final_Elbow.ipynb
    ├── EfficientNetB1_Final_Hand.ipynb
    ├── EfficientNetB1_Final_Shoulder.ipynb
    └── EfficientNetB1_Parts.ipynb

Install dependencies
pip install -r requirements.txt
▶️ Run the Application
python app.py

Then open in browser:

http://127.0.0.1:5000
📈 Future Improvements

Add more anatomical regions

Support DICOM images

Add uncertainty estimation

Integrate model monitoring for drift detection

Improve UI/UX for clinical deployment

🎯 Applications

Clinical decision support

Radiology workflow assistance

Medical imaging research

AI in healthcare education

📜 License

This project is intended for educational and research purposes.

👨‍💻 Author

Developed as part of an AI/ML research project focused on medical image analysis and deployment-ready deep learning systems.

⭐ If you found this project useful, consider giving it a star!
