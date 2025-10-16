from flask import Blueprint, request, jsonify
from transformers import pipeline

sentiment_bp = Blueprint("sentiment_bp", __name__)

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="w11wo/indonesian-roberta-base-sentiment-classifier",
    top_k=1
)

@sentiment_bp.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "API Sentimen Interview Feedback ✅",
        "usage": "POST ke /sentiment/analyze dengan JSON { 'feedback': '...' }"
    })

@sentiment_bp.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data or "feedback" not in data:
        return jsonify({"error": "Masukkan JSON dengan key 'feedback'"}), 400

    feedback = data["feedback"]

    result = sentiment_analyzer(feedback, top_k=1)

    if isinstance(result, list) and isinstance(result[0], list):
        result = result[0][0]
    elif isinstance(result, list):
        result = result[0]

    response = {
        "feedback": feedback,
        "label": result["label"],
        "score": round(float(result["score"]), 4)
    }

    return jsonify(response)
