from flask import Flask, request, jsonify
from PIL import Image
import google.generativeai as genai
import tempfile
import os
import json

# ✅ Use API key from environment variable
genai.configure(api_key=os.environ["GENAI_API_KEY"])

# ✅ Initialize Gemini model
model = genai.GenerativeModel("models/gemini-1.5-flash")

# ✅ Prompt
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

# ✅ Flask app
app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        image_path = tmp.name
        image_file.save(image_path)

    try:
        image = Image.open(image_path)
        response = model.generate_content([PROMPT, image])
        try:
            result = json.loads(response.text)
            return jsonify(result)
        except json.JSONDecodeError:
            return jsonify({"error": "Gemini output not JSON", "raw_response": response.text}), 500
    finally:
        os.remove(image_path)

# ✅ Run server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
