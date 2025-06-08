from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import google.generativeai as genai
import tempfile
import os
import json
import pandas as pd
import joblib
import shap
import numpy as np
import requests
import cv2
from pyzbar.pyzbar import decode

# Configure Gemini
genai.configure(api_key=os.environ["GENAI_API_KEY"])

# Load model pipeline
model_path = "food_consumption_model_xgb.pkl"
xgb_pipeline = joblib.load(model_path)

# Prompt for Gemini
PROMPT = """
You are an AI assistant that extracts structured food product data from packaging images.

Please extract:
1. A list of ingredients.
2. Nutrition facts (key: value, with units).

Return the result in JSON format like this:
{
  "ingredients": [ ... ],
  "nutrition_facts": {
    "Calories": "...",
    "Total Fat": "...",
    ...
  }
}
"""

# Flask setup
app = Flask(__name__)
CORS(app, supports_credentials=True, origins="*")

def flatten(value):
    if isinstance(value, list):
        return ', '.join(map(str, value))
    return value

def query_openfoodfacts(barcode):
    url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data.get("status") == 1:
            product = data["product"]
            return {
                "ingredients": product.get("ingredients_text", "").split(", "),
                "allergens": product.get("allergens_tags", []),
                "nutrition_facts": product.get("nutriments", {})
            }
    return None

def extract_barcode_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    barcodes = decode(image)
    if barcodes:
        return barcodes[0].data.decode('utf-8')
    return None

def custom_nutriscore(user_profile, ingredients, allergens):
    score = 100  # Start from full score
    deductions = []

    # Allergen-based deduction
    user_allergens = [a.lower().strip() for a in user_profile.get("allergies", "").split(",") if a != 'none']
    product_allergens = [a.split(":")[-1].replace("_", " ").lower().strip() for a in allergens]
    for allergen in user_allergens:
        if allergen in product_allergens:
            score -= 30
            deductions.append(f"❌ Allergen conflict: {allergen}")

    # Vegan diet check
    if user_profile.get("diet", "").lower().strip() == "vegan":
        animal_keywords = [
            "milk", "egg", "honey", "gelatin", "beef", "chicken", "fish", "meat",
            "lard", "casein", "lactose", "whey", "shellfish", "anchovy", "animal"
        ]
        for ingredient in ingredients:
            for keyword in animal_keywords:
                if keyword in ingredient.lower():
                    score -= 20
                    deductions.append(f"🐄 Animal-based ingredient found: {ingredient}")
                    break

    return max(score, 0), deductions

@app.route('/analyze', methods=['POST'])
def analyze_image():
    print("✅ /analyze request received")

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    profile_data = request.form.get("profile")
    barcode = request.form.get("barcode", "").strip()

    user_profile = {
        "allergies": "none",
        "diet": "none",
        "conditions": "none"
    }

    if profile_data:
        try:
            user_profile = json.loads(profile_data)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON in profile"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image_path = tmp.name
        image_file.save(image_path)

    try:
        if not barcode:
            barcode = extract_barcode_from_image(image_path)
            if barcode:
                print(f"🔍 Extracted barcode: {barcode}")

        extracted = None
        if barcode:
            off_data = query_openfoodfacts(barcode)
            if off_data:
                extracted = {
                    "ingredients": off_data["ingredients"],
                    "nutrition_facts": {
                        "Calories": off_data["nutrition_facts"].get("energy-kcal_100g", 0),
                        "Total Fat": off_data["nutrition_facts"].get("fat_100g", 0),
                        "Saturated Fat": off_data["nutrition_facts"].get("saturated-fat_100g", 0),
                        "Sodium": off_data["nutrition_facts"].get("sodium_100g", 0),
                        "Total Carbohydrate": off_data["nutrition_facts"].get("carbohydrates_100g", 0),
                        "Sugar": off_data["nutrition_facts"].get("sugars_100g", 0),
                        "Protein": off_data["nutrition_facts"].get("proteins_100g", 0)
                    },
                    "allergens": off_data["allergens"]
                }

        if not extracted:
            image = Image.open(image_path)
            response = genai.GenerativeModel("models/gemini-1.5-flash").generate_content([PROMPT, image])
            clean_text = response.text.strip().strip("```json").strip("```").strip()

            try:
                extracted = json.loads(clean_text)
                extracted["allergens"] = []  # No allergen info from Gemini
            except json.JSONDecodeError:
                return jsonify({"error": "Gemini output not JSON parseable", "raw_response": response.text}), 500

        # Prepare input for ML model
        input_data = {
            "allergies": flatten(user_profile.get("allergies", "none")),
            "diet": flatten(user_profile.get("diet", "none")),
            "conditions": flatten(user_profile.get("conditions", "none")),
            "ingredients": flatten(extracted.get("ingredients", []))
        }

        nutrition = extracted.get("nutrition_facts", {})
        for col in ['Calories', 'Total Fat', 'Saturated Fat', 'Sodium',
                    'Total Carbohydrate', 'Sugar', 'Protein']:
            val = nutrition.get(col, "0")
            try:
                input_data[col] = float(val.split()[0]) if isinstance(val, str) else float(val)
            except Exception:
                input_data[col] = 0.0

        df_input = pd.DataFrame([input_data])
        for col in df_input.columns:
            if isinstance(df_input[col].iloc[0], list):
                df_input[col] = df_input[col].apply(flatten)

        prediction = xgb_pipeline.predict(df_input)[0]
        prediction_str = "Yes" if prediction == "Yes" else "No"

        model = xgb_pipeline.named_steps["classifier"]
        preprocessor = xgb_pipeline.named_steps["preprocessor"]
        X_transformed = preprocessor.transform(df_input)

        explainer = shap.Explainer(model)
        shap_values = explainer(X_transformed)

        try:
            feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
            feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

        contributions = shap_values.values[0]
        feature_contributions = list(zip(feature_names, contributions))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        explanations = []
        for feature, value in feature_contributions[:8]:
            direction = "🔴" if value < -0.01 else "🟢" if value > 0.01 else "🟡"
            explanation = f"{direction} {feature} → {'Negative' if value < -0.01 else 'Positive' if value > 0.01 else 'Neutral'}"
            explanations.append(explanation)

        nutriscore, deduction_reasons = custom_nutriscore(user_profile, extracted.get("ingredients", []), extracted.get("allergens", []))

        return jsonify({
            "profile": user_profile,
            "source": "OpenFoodFacts" if barcode else "Gemini",
            "analysis": extracted,
            "should_consume": prediction_str,
            "ml_explanation": explanations,
            "custom_nutriscore": nutriscore,
            "deductions": deduction_reasons
        })

    finally:
        os.remove(image_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
