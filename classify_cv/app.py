# classify_cv/app.py
from flask import Blueprint, request, jsonify
import joblib
import os
from classify_cv.preprocess import preprocess_cv
from PyPDF2 import PdfReader

cv_bp = Blueprint("cv_bp", __name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model_path = os.path.join(BASE_DIR, "model_st.joblib")
clf, encoder = joblib.load(model_path)

@cv_bp.route("/classify", methods=["POST"])
def classify_cv():
    if "file" not in request.files:
        return jsonify({"error": "Harap unggah file PDF menggunakan key 'file'"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Hanya file PDF yang diterima"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    cleaned_text = preprocess_cv(text)

    X_vec = encoder.encode([cleaned_text])
    prediction = clf.predict(X_vec)[0]
    proba = clf.predict_proba(X_vec)[0]
    confidence = dict(zip(clf.classes_, [round(float(p), 4) for p in proba]))

    return jsonify({
        "filename": file.filename,
        "prediction": prediction,
        "confidence": confidence
    })