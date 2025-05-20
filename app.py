from flask import Flask, request, jsonify
from PIL import Image
import google.generativeai as genai
import json
import tempfile
import os

app = Flask(__name__)

# ✅ Configure Gemini
genai.configure(api_key=os.getenv("GENAI_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-flash")

@app.route("/analyze", methods=["POST"])
def analyze():
    image_file = request.files.get("image")
    if not image_file:
        return jsonify({"error": "No image uploaded"}), 400

    # ✅ Step 1: Get user profile JSON from frontend (optional)
    profile_json = request.form.get("profile")
    user_profile = {}

    if profile_json:
        try:
            user_profile = json.loads(profile_json)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON in 'profile'"}), 400

    # ✅ Step 2: Save the uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image_path = temp_file.name
        image_file.save(image_path)

    try:
        image = Image.open(image_path)

        # ✅ Step 3: Prompt Gemini
        prompt = """
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
        response = model.generate_content([prompt, image])
        extracted_data = json.loads(response.text)

        # ✅ Step 4: Return extracted data along with profile (no matching yet)
        return jsonify({
            "extracted_data": extracted_data,
            "user_profile": user_profile  # <-- returned for now, not used
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        os.remove(image_path)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
