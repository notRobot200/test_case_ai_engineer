# ranking_canditates/app.py
from flask import Blueprint, request, jsonify
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import os

rank_bp = Blueprint("rank_bp", __name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + " "
    return text

@rank_bp.route("/rank-candidates", methods=["POST"])
def rank_candidates():
    if "job_description" not in request.files:
        return jsonify({"error": "Job description file required"}), 400

    job_desc_file = request.files["job_description"]
    job_desc_path = os.path.join(UPLOAD_FOLDER, job_desc_file.filename)
    job_desc_file.save(job_desc_path)
    with open(job_desc_path, "r", encoding="utf-8") as f:
        job_description = f.read()

    cv_files = request.files.getlist("cvs")
    if not cv_files:
        return jsonify({"error": "At least one CV file required"}), 400

    cv_texts, cv_names = [], []
    for cv_file in cv_files:
        cv_path = os.path.join(UPLOAD_FOLDER, cv_file.filename)
        cv_file.save(cv_path)
        text = extract_text_from_pdf(cv_path)
        cv_texts.append(text)
        cv_names.append(cv_file.filename)

    jd_embedding = model.encode(job_description, convert_to_tensor=True)
    cv_embeddings = model.encode(cv_texts, convert_to_tensor=True)
    similarities = util.cos_sim(jd_embedding, cv_embeddings).cpu().numpy().flatten()

    candidates = [{"cv": name, "score": float(score)} for name, score in zip(cv_names, similarities)]
    candidates_sorted = sorted(candidates, key=lambda x: x["score"], reverse=True)

    return jsonify({"ranking": candidates_sorted})
