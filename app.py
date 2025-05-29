from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import google.generativeai as genai
import tempfile
import os
import json
import pandas as pd
import joblib

# ✅ Use your Gemini API key from environment variable
genai.configure(api_key=os.environ["GENAI_API_KEY"])

# ✅ Load the trained model pipeline
model_path = "food_consumption_model_xgb.pkl"
xgb_pipeline = joblib.load(model_path)

# ✅ Prompt for Gemini
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

# ✅ Initialize Flask app
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

    # Save the image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image_path = tmp.name
        image_file.save(image_path)

    try:
        # ✅ Send image to Gemini
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

        # ✅ Default profile if none
        if not user_profile:
            user_profile = {
                "allergies": "none",
                "diet": "none",
                "conditions": "none"
            }

        # ✅ Build input data
        input_data = {
            "allergies": user_profile.get("allergies", "none"),
            "diet": user_profile.get("diet", "none"),
            "conditions": user_profile.get("conditions", "none"),
            "ingredients": ','.join(extracted.get("ingredients", []))
        }

        # ✅ Parse numerical nutrition facts
        nutrition = extracted.get("nutrition_facts", {})
        for col in ['Calories', 'Total Fat', 'Saturated Fat', 'Sodium',
                    'Total Carbohydrate', 'Sugar', 'Protein']:
            val = nutrition.get(col, "0")
            try:
                input_data[col] = float(val.split()[0]) if isinstance(val, str) else float(val)
            except Exception:
                input_data[col] = 0.0  # fallback if parsing fails

        df_input = pd.DataFrame([input_data])

        # ✅ Run prediction using pipeline (which includes preprocessing)
        prediction = xgb_pipeline.predict(df_input)[0]
        prediction_str = "Yes" if prediction == "Yes" else "No"

        return jsonify({
            "profile": user_profile,
            "analysis": extracted,
            "should_consume": prediction_str
        })

    finally:
        os.remove(image_path)


# ✅ Dev server (use Gunicorn in production)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
