import os
import uuid
import datetime
from datetime import timezone  
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from pymongo import MongoClient, server_api  
from pymongo.server_api import ServerApi
from src.pipeline.predict_pipeline import PredictPipeline
from src.logger import logger
import certifi
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

MONGO_URI = os.getenv("MONGODB_URL_KEY") 

if not MONGO_URI:
    logger.error("🚨 Critical Error: 'MONGODB_URL_KEY' environment variable not found!")
    MONGO_URI = "mongodb://localhost:27017/"

try:
    client = MongoClient(
        MONGO_URI, 
        tlsCAFile=certifi.where(),
        server_api=server_api.ServerApi("1")
    )
    db = client["food_detector_db"]         
    
    records_collection = db["predictions"] 
    print("✅ [SUCCESS] MongoDB Successfully Connected!")
    logger.info("MongoDB connected successfully using MONGODB_URL_KEY.")
except Exception as e:
    records_collection = None
    print(f"🚨 [ERROR] MongoDB Connection Failed: {e}")
    logger.error(f"MongoDB connection error: {e}")

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

def get_detected_food_nutrition(top5_predictions: list) -> dict:
    foods_list = []
    total_calories = 0
    total_sugar = 0.0
    total_protein = 0.0
    total_fiber = 0.0

    for label, confidence in top5_predictions:
        if confidence >= 0.5:  
            key = label.strip()
            info = NUTRITION_DB.get(key)
            conf_percent = round(confidence * 100, 1)

            if info:
                foods_list.append({
                    "name":       key,
                    "confidence": conf_percent,
                    "calories":   info["calories"],
                    "sugar":      info["sugar"],
                    "protein":    info["protein"],
                    "fiber":      info["fiber"],
                    "serving":    info["serving"],
                })
                total_calories += info["calories"]
                total_sugar += info["sugar"]
                total_protein += info["protein"]
                total_fiber += info["fiber"]
            else:
                foods_list.append({
                    "name":       key,
                    "confidence": conf_percent,
                    "calories":   "N/A", "sugar": "N/A", "protein": "N/A", "fiber": "N/A", "serving": "N/A"
                })

    return {
        "foods": foods_list,
        "totals": {
            "calories": total_calories,
            "sugar":    round(total_sugar, 1),
            "protein":  round(total_protein, 1),
            "fiber":    round(total_fiber, 1)
        }
    }

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    predictor = PredictPipeline()
    user_name = request.form.get("user_name", "Anonymous").strip() or "Anonymous"
    
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use PNG / JPG / JPEG / WEBP"}), 400

    try:
        ext      = file.filename.rsplit(".", 1)[1].lower()
        unique_id = uuid.uuid4().hex
        filename = secure_filename(f"{unique_id}.{ext}")
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        labels, top5 = predictor.predict(filepath)
        image_url    = f"/static/uploads/{filename}"

        if not top5:
            return jsonify({"error": "No food detected"}), 400
            
        nutrition = get_detected_food_nutrition(top5)

        if not nutrition["foods"]:
            return jsonify({"error": "Detected foods are below 50% confidence threshold"}), 400

        document_id = f"req_{unique_id[:12]}"
        
        current_utc_time = datetime.datetime.now(timezone.utc).isoformat()
        
        mongo_document = {
            "_id": document_id,                    
            "user_name": user_name,                
            "image_name": filename,
            "image_url": image_url,
            "detected_labels": [f["name"] for f in nutrition["foods"]],
            "nutrition_details": nutrition["foods"],
            "total_summary": nutrition["totals"],
            "timestamp": current_utc_time
        }
        
        if records_collection is not None:
            try:
                records_collection.insert_one(mongo_document)
                print(f"✅ [SUCCESS] Data saved to MongoDB! ID: {document_id}")
                logger.info(f"Successfully stored record {document_id} via MONGODB_URL_KEY")
            except Exception as db_err:
                print(f"🚨 [MONGODB WRITE ERROR] Failed to save data: {db_err}")
                logger.error(f"MongoDB Insert failure: {db_err}")
        else:
            print("🚨 [MONGODB SKIPPED] Collection is not active (check your MONGODB_URL_KEY).")

        return jsonify({
            "success":   True,
            "id":        document_id,
            "user_name": user_name,
            "labels":    mongo_document["detected_labels"],
            "image_url": image_url,
            "nutrition": nutrition
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False,host="0.0.0.0", port=5000)