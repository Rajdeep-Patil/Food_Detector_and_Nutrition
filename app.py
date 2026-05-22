import os
import uuid
import datetime
from datetime import timezone
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from pymongo import MongoClient
import certifi
from dotenv import load_dotenv

from src.pipeline.predict_pipeline import PredictPipeline
from src.logger import logger

load_dotenv()

app = Flask(__name__)

# ---------------- CONFIG ----------------
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ---------------- MONGO ----------------
MONGO_URI = os.getenv("MONGODB_URL_KEY")

if not MONGO_URI:
    logger.error("MONGODB_URL_KEY not found")
    MONGO_URI = "mongodb://localhost:27017/"

try:
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    db = client["food_detector_db"]
    records_collection = db["predictions"]
    print("✅ MongoDB Connected")
except Exception as e:
    records_collection = None
    print("❌ MongoDB Failed:", e)

# ---------------- LAZY MODEL LOADING (IMPORTANT) ----------------
predictor = None

# ---------------- NUTRITION DB ----------------
NUTRITION_DB = {
    "AW cola": {"calories": 150, "sugar": 39.0, "protein": 0.0, "fiber": 0.0, "serving": "355 ml can"},
    "Beijing Beef": {"calories": 470, "sugar": 25.0, "protein": 26.0, "fiber": 3.0, "serving": "1 serving (~340 g)"},
    "Chow Mein": {"calories": 490, "sugar": 8.0, "protein": 22.0, "fiber": 4.0, "serving": "1 serving (~283 g)"},
    "Fried Rice": {"calories": 530, "sugar": 3.0, "protein": 16.0, "fiber": 2.0, "serving": "1 serving (~283 g)"},
    "Hashbrown": {"calories": 140, "sugar": 0.5, "protein": 2.0, "fiber": 1.5, "serving": "1 piece (~75 g)"},
    "Honey Walnut Shrimp": {"calories": 360, "sugar": 18.0, "protein": 13.0, "fiber": 1.0, "serving": "1 serving (~170 g)"},
    "Kung Pao Chicken": {"calories": 290, "sugar": 13.0, "protein": 22.0, "fiber": 2.0, "serving": "1 serving (~170 g)"},
    "String Bean Chicken Breast": {"calories": 190, "sugar": 9.0, "protein": 14.0, "fiber": 3.0, "serving": "1 serving (~170 g)"},
    "Super Greens": {"calories": 90, "sugar": 4.0, "protein": 6.0, "fiber": 5.0, "serving": "1 serving (~170 g)"},
    "The Original Orange Chicken": {"calories": 490, "sugar": 19.0, "protein": 25.0, "fiber": 1.0, "serving": "1 serving (~170 g)"},
    "White Steamed Rice": {"calories": 380, "sugar": 0.0, "protein": 7.0, "fiber": 0.5, "serving": "1 serving (~280 g)"},
    "black pepper rice bowl": {"calories": 520, "sugar": 6.0, "protein": 28.0, "fiber": 2.0, "serving": "1 bowl (~350 g)"},
    "burger": {"calories": 540, "sugar": 8.0, "protein": 28.0, "fiber": 2.0, "serving": "1 burger (~200 g)"},
    "carrot_eggs": {"calories": 180, "sugar": 5.0, "protein": 10.0, "fiber": 2.5, "serving": "1 serving (~150 g)"},
    "cheese burger": {"calories": 620, "sugar": 9.0, "protein": 32.0, "fiber": 2.0, "serving": "1 burger (~220 g)"},
    "chicken waffle": {"calories": 560, "sugar": 12.0, "protein": 30.0, "fiber": 2.0, "serving": "1 serving (~250 g)"},
    "chicken_nuggets": {"calories": 470, "sugar": 1.0, "protein": 28.0, "fiber": 1.0, "serving": "10 pieces (~170 g)"},
    "chinese_cabbage": {"calories": 25, "sugar": 2.0, "protein": 2.0, "fiber": 2.5, "serving": "1 cup (~100 g)"},
    "chinese_sausage": {"calories": 320, "sugar": 5.0, "protein": 16.0, "fiber": 0.0, "serving": "2 links (~80 g)"},
    "crispy corn": {"calories": 210, "sugar": 4.0, "protein": 4.0, "fiber": 3.0, "serving": "1 serving (~80 g)"},
    "curry": {"calories": 310, "sugar": 7.0, "protein": 20.0, "fiber": 4.0, "serving": "1 serving (~250 g)"},
    "french fries": {"calories": 365, "sugar": 0.5, "protein": 4.0, "fiber": 3.5, "serving": "medium (~117 g)"},
    "fried chicken": {"calories": 400, "sugar": 1.0, "protein": 32.0, "fiber": 0.5, "serving": "1 piece (~180 g)"},
    "fried_chicken": {"calories": 400, "sugar": 1.0, "protein": 32.0, "fiber": 0.5, "serving": "1 piece (~180 g)"},
    "fried_dumplings": {"calories": 330, "sugar": 2.0, "protein": 14.0, "fiber": 2.0, "serving": "6 pieces (~150 g)"},
    "fried_eggs": {"calories": 185, "sugar": 0.5, "protein": 13.0, "fiber": 0.0, "serving": "2 eggs (~100 g)"},
    "mango chicken pocket": {"calories": 480, "sugar": 14.0, "protein": 24.0, "fiber": 2.0, "serving": "1 pocket (~200 g)"},
    "mozza burger": {"calories": 670, "sugar": 10.0, "protein": 35.0, "fiber": 2.0, "serving": "1 burger (~240 g)"},
    "mung_bean_sprouts": {"calories": 30, "sugar": 2.0, "protein": 3.0, "fiber": 2.0, "serving": "1 cup (~104 g)"},
    "nugget": {"calories": 280, "sugar": 1.0, "protein": 17.0, "fiber": 0.5, "serving": "6 pieces (~100 g)"},
    "perkedel": {"calories": 220, "sugar": 1.5, "protein": 7.0, "fiber": 2.0, "serving": "2 pieces (~100 g)"},
    "rice": {"calories": 206, "sugar": 0.0, "protein": 4.3, "fiber": 0.6, "serving": "1 cup cooked (~186 g)"},
    "sprite": {"calories": 140, "sugar": 38.0, "protein": 0.0, "fiber": 0.0, "serving": "355 ml can"},
    "tostitos cheese dip sauce": {"calories": 180, "sugar": 3.0, "protein": 4.0, "fiber": 1.0, "serving": "4 tbsp (~60 g)"},
    "triangle_hash_brown": {"calories": 150, "sugar": 0.5, "protein": 2.0, "fiber": 1.5, "serving": "1 piece (~80 g)"},
    "water_spinach": {"calories": 20, "sugar": 1.0, "protein": 2.5, "fiber": 2.0, "serving": "1 cup (~56 g)"},
}

# ---------------- HELPERS ----------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_nutrition(top5):
    foods = []
    totals = {"calories": 0, "sugar": 0, "protein": 0, "fiber": 0}

    for label, score in top5:
        if score < 0.5:
            continue

        info = NUTRITION_DB.get(label.strip())
        conf = round(score * 100, 1)

        if info:
            foods.append({
                "name": label,
                "confidence": conf,
                "calories": info["calories"],
                "sugar": info["sugar"],
                "protein": info["protein"],
                "fiber": info["fiber"],
                "serving": info["serving"],
            })

            totals["calories"] += info["calories"]
            totals["sugar"] += info["sugar"]
            totals["protein"] += info["protein"]
            totals["fiber"] += info["fiber"]

    return {"foods": foods, "totals": totals}


# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    global predictor

    user_name = request.form.get("user_name", "Anonymous").strip() or "Anonymous"

    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]

    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    try:
        # ---------------- SAVE IMAGE ----------------
        ext = file.filename.rsplit(".", 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(filename))
        file.save(filepath)

        # ---------------- LAZY LOAD MODEL ----------------
        if predictor is None:
            predictor = PredictPipeline()

        # ---------------- PREDICT ----------------
        labels, top5 = predictor.predict(filepath)

        if not top5:
            return jsonify({"error": "No prediction"}), 400

        nutrition = get_nutrition(top5)

        if not nutrition["foods"]:
            return jsonify({"error": "Low confidence"}), 400

        image_url = f"/static/uploads/{filename}"
        req_id = f"req_{uuid.uuid4().hex[:10]}"

        document = {
            "_id": req_id,
            "user_name": user_name,
            "image_url": image_url,
            "detected_labels": [f["name"] for f in nutrition["foods"]],
            "nutrition": nutrition,
            "timestamp": datetime.datetime.now(timezone.utc).isoformat()
        }

        if records_collection:
            records_collection.insert_one(document)

        return jsonify({
            "success": True,
            "id": req_id,
            "labels": document["detected_labels"],
            "image_url": image_url,
            "nutrition": nutrition
        })

    except Exception as e:
        logger.error(str(e))
        return jsonify({"error": str(e)}), 500


# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)