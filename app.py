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

@app.route('/analyze', methods=['POST'])
def analyze_image():
    print("✅ /analyze request received")

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    profile_data = request.form.get("profile")

    user_profile = None
    if profile_data:
        try:
            user_profile = json.loads(profile_data)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON in profile"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image_path = tmp.name
        image_file.save(image_path)

    try:
        # Step 1: Extract text from image via Gemini
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

        if not user_profile:
            user_profile = {
                "allergies": "none",
                "diet": "none",
                "conditions": "none"
            }

        # Flatten helper
        def flatten(value):
            if isinstance(value, list):
                return ', '.join(map(str, value))
            return value

        # Prepare input data
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

        # Fix list columns
        for col in df_input.columns:
            if isinstance(df_input[col].iloc[0], list):
                df_input[col] = df_input[col].apply(flatten)

        # Predict
        prediction = xgb_pipeline.predict(df_input)[0]
        prediction_str = "Yes" if prediction == "Yes" else "No"

        # SHAP explanation
        model = xgb_pipeline.named_steps["model"]
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

        return jsonify({
            "profile": user_profile,
            "analysis": extracted,
            "should_consume": prediction_str,
            "explanation": explanations
        })

    finally:
        os.remove(image_path)

# Run server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
