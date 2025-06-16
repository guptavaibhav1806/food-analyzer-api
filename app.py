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
from pyNutriScore import NutriScore

# Configure Gemini
genai.configure(api_key=os.environ["GENAI_API_KEY"])

# Load model pipeline
model_path = "food_consumption_model_xgb.pkl"
xgb_pipeline = joblib.load(model_path)

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

app = Flask(__name__)
CORS(app, supports_credentials=True, origins="*")

def flatten(value):
    if isinstance(value, list):
        return ', '.join(str(v).strip() for v in value)
    return str(value).strip()

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
                "nutrition_facts": product.get("nutriments", {}),
                "nutriscore_data": {
                    "score": product.get("nutriscore_score"),
                    "grade": product.get("nutriscore_grade"),
                }
            }
    return None

def compute_pynutriscore(nutrition_facts):
    try:
        score = NutriScore().calculate({
            'energy': nutrition_facts.get("energy-kcal_100g", 0),
            'fibers': nutrition_facts.get("fiber_100g", 0),
            'fruit_percentage': nutrition_facts.get("fruits-vegetables-nuts_100g", 0),
            'proteins': nutrition_facts.get("proteins_100g", 0),
            'saturated_fats': nutrition_facts.get("saturated-fat_100g", 0),
            'sodium': nutrition_facts.get("sodium_100g", 0),
            'sugar': nutrition_facts.get("sugars_100g", 0),
        }, 'solid')
        grade = NutriScore().calculate_class({
            'energy': nutrition_facts.get("energy-kcal_100g", 0),
            'fibers': nutrition_facts.get("fiber_100g", 0),
            'fruit_percentage': nutrition_facts.get("fruits-vegetables-nuts_100g", 0),
            'proteins': nutrition_facts.get("proteins_100g", 0),
            'saturated_fats': nutrition_facts.get("saturated-fat_100g", 0),
            'sodium': nutrition_facts.get("sodium_100g", 0),
            'sugar': nutrition_facts.get("sugars_100g", 0),
        }, 'solid')
        return score, grade
    except Exception:
        return None, None

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    profile_data = request.form.get("profile")
    barcode = request.form.get("barcode", "").strip()

    user_profile = {
        "allergies": [],
        "diet": "none",
        "conditions": []
    }

    if profile_data:
        try:
            profile_json = json.loads(profile_data)
            user_profile["allergies"] = profile_json.get("allergies", [])
            user_profile["diet"] = profile_json.get("diet", "none")
            user_profile["conditions"] = profile_json.get("conditions", [])
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON in profile"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image_path = tmp.name
        image_file.save(image_path)

    try:
        extracted = None
        nutriscore_score = None
        nutriscore_grade = None

        if barcode:
            off_data = query_openfoodfacts(barcode)
            if off_data:
                user_allergies = [a.lower().strip() for a in user_profile.get("allergies", [])]
                product_allergens = [a.split(":")[-1].replace("_", " ").lower().strip() for a in off_data.get("allergens", [])]

                if any(allergen in product_allergens for allergen in user_allergies):
                    return jsonify({
                        "profile": user_profile,
                        "source": "OpenFoodFacts",
                        "analysis": {
                            "ingredients": off_data["ingredients"],
                            "nutrition_facts": off_data["nutrition_facts"],
                            "allergens": product_allergens
                        },
                        "should_consume": "No",
                        "nutriscore": {
                            "score": 30,
                            "grade": "D"
                        },
                        "reason": "Allergen conflict detected from OpenFoodFacts"
                    })

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
                    }
                }

                nutriscore_score = off_data["nutriscore_data"].get("score")
                nutriscore_grade = off_data["nutriscore_data"].get("grade")

                if nutriscore_score is None or nutriscore_grade in (None, "unknown"):
                    nutriscore_score, nutriscore_grade = compute_pynutriscore(off_data["nutrition_facts"])

        if not extracted:
            image = Image.open(image_path)
            response = genai.GenerativeModel("models/gemini-1.5-flash").generate_content([PROMPT, image])
            clean_text = response.text.strip().strip("```json").strip("```").strip()
            try:
                extracted = json.loads(clean_text)
            except json.JSONDecodeError:
                return jsonify({
                    "error": "Gemini output not JSON parseable",
                    "raw_response": response.text
                }), 500

        if nutriscore_score is None or nutriscore_grade is None:
            nutriscore_score = 100
            ingredients = [i.lower() for i in extracted.get("ingredients", [])]
            if user_profile.get("diet", "").lower() == "vegan":
                if any(word in ing for ing in ingredients for word in ["milk", "egg", "honey", "gelatin", "meat", "fish"]):
                    nutriscore_score -= 30
            user_allergies = [a.lower().strip() for a in user_profile.get("allergies", [])]
            if any(a in ing for a in user_allergies for ing in ingredients):
                nutriscore_score -= 30

            nutriscore_grade = "A" if nutriscore_score >= 80 else "B" if nutriscore_score >= 60 else "C" if nutriscore_score >= 40 else "D" if nutriscore_score >= 20 else "E"

        return jsonify({
            "profile": user_profile,
            "source": "OpenFoodFacts" if barcode else "Gemini",
            "analysis": extracted,
            "should_consume": "Yes" if nutriscore_score > 30 else "No",
            "nutriscore": {
                "score": nutriscore_score,
                "grade": nutriscore_grade
            }
        })
    finally:
        os.remove(image_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
