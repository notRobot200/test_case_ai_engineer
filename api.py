from flask import Flask
from ranking_candidates.app import rank_bp
from classify_cv.app import cv_bp
from sentiment.app import sentiment_bp

app = Flask(__name__)

app.register_blueprint(rank_bp, url_prefix="/ranking")
app.register_blueprint(cv_bp, url_prefix="/classify")
app.register_blueprint(sentiment_bp, url_prefix="/sentiment")

if __name__ == "__main__":
    app.run(debug=True)
