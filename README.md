# Food Analysis and Nutrition Chatbot API

This project provides a Flask-based API with two main functionalities:

1. **Food Product Analysis**: Extracts ingredients and nutritional information from food packaging images or barcodes. It then analyzes this data against a user's profile (allergies, diet, health conditions) to provide a "should consume" recommendation and a Nutri-Score.
2. **Nutrition Chatbot**: A conversational AI that acts as a nutrition expert, providing personalized food advice based on a user's profile.

The application leverages the Google Gemini API for image analysis and chat responses, Open Food Facts for barcode data, and a pre-trained XGBoost model for further analysis.

## Features

- **Image-based Food Analysis**: Upload a picture of a food product's packaging to extract ingredients and nutrition facts.
- **Barcode Scanner Integration**: Fetch product data directly from the Open Food Facts database using a barcode.
- **Personalized Recommendations**: Get a "Yes/No" consumption recommendation based on your allergies, dietary restrictions (e.g., vegan), and health conditions.
- **Nutri-Score Calculation**: Computes a Nutri-Score and grade for the product.
- **AI-Powered Nutrition Chat**: Ask food-related questions and get personalized advice from an AI expert.
- **Fallback System**: If barcode data is unavailable, it falls back to analyzing the uploaded image.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- Python 3.8+
- pip (Python package installer)

## Setup and Installation

Follow these steps to get the application running on your local machine.

### 1. Clone the Repository

First, clone this repository to your local machine.

```bash
git clone https://github.com/LifeAtlas/life-atlas-food-analyzer-api.git
cd life-atlas-food-analyzer-api
```

### 2. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

#### For macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

#### For Windows
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

Install all the required Python packages using the requirements.txt file.

```bash
pip install -r requirements.txt
```

**Note:** If a requirements.txt file is not provided, you can install the packages manually:

```bash
pip install Flask Flask-Cors Pillow google-generativeai pandas joblib requests pyNutriScore scikit-learn xgboost
```

### 4. Configure Environment Variables (for Local Development)

The application requires an API key for the Google Gemini API. For local development, you need to set this as an environment variable.

Create a `.env` file in the root directory of the project and add your API key:

```
GENAI_API_KEY="YOUR_GEMINI_API_KEY"
```

The application will load this key automatically when running locally. **Do not commit this file to version control.**

#### To obtain a Google Gemini API key:

1. Go to the [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click on "Get API key" and create a new key
4. Copy the key and paste it into your `.env` file for local testing or into your hosting service's secret manager for deployment

### 5. Download the Machine Learning Model

The application uses a pre-trained XGBoost model (`food_consumption_model_xgb.pkl`). Make sure this file is present in the root directory of the project. If you have the model stored elsewhere, update the `model_path` variable in `app.py`.

## Running the Application Locally

Once the setup is complete, you can start the Flask server on your local machine.

```bash
python app.py
```

The application will start on `http://0.0.0.0:5000`. The `debug=True` flag enables hot-reloading, so the server will restart automatically when you make changes to the code.

## Deployment on Render

This project includes a `render.yaml` file, which allows for easy deployment to the Render cloud platform.

1. Fork this repository to your GitHub account
2. Go to the [Render Dashboard](https://dashboard.render.com/) and create a new "Blueprint" service
3. Connect the forked repository. Render will automatically detect and use the `render.yaml` file for configuration
4. **Set the Environment Variable**: In the service settings on Render, you must add the `GENAI_API_KEY` as an environment variable:
   - **Key**: `GENAI_API_KEY`
   - **Value**: `Your_Actual_Gemini_API_Key`
5. Render will build and deploy the application. The service will be available at the URL provided by Render

## API Endpoints

The application exposes two API endpoints.

### 1. `/analyze`

This endpoint analyzes a food product based on an image or barcode and a user's profile.

- **Method**: `POST`
- **Content-Type**: `multipart/form-data`

#### Request Form Data:

- `image` (file, optional): An image file of the food product's packaging
- `barcode` (string, optional): The barcode of the product
- `profile` (string, optional): A JSON string representing the user's profile
  - `allergies` (array of strings)
  - `diet` (string, e.g., "vegan", "none")
  - `conditions` (array of strings)

#### Example profile JSON string:

```json
{
  "allergies": ["nuts", "soy"],
  "diet": "vegan",
  "conditions": ["high blood pressure"]
}
```

#### Example curl Request:

```bash
curl -X POST \
  -F "image=@/path/to/your/image.jpg" \
  -F 'profile={"allergies":["milk"],"diet":"none","conditions":[]}' \
  http://127.0.0.1:5000/analyze
```

#### Success Response (200 OK):

```json
{
    "analysis": {
        "ingredients": ["..."],
        "nutrition_facts": {
            "Calories": "...",
            "Total Fat": "...",
            "..."
        }
    },
    "nutriscore": {
        "grade": "A",
        "score": 90
    },
    "profile": {
        "allergies": ["milk"],
        "diet": "none",
        "conditions": []
    },
    "should_consume": "Yes",
    "source": "Gemini"
}
```

#### Error Response (400/500):

```json
{
    "error": "Error message explaining the issue."
}
```

### 2. `/chat`

This endpoint provides a conversational interface with the nutrition expert chatbot.

- **Method**: `POST`
- **Content-Type**: `application/json`

#### Request Body (JSON):

- `message` (string, required): The user's message or question
- `profile` (object, optional): A JSON object with the user's profile (same structure as for `/analyze`)

#### Example curl Request:

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"message": "What are some good sources of protein for a vegan diet?", "profile": {"diet": "vegan"}}' \
  http://127.0.0.1:5000/chat
```

#### Success Response (200 OK):

```json
{
    "response": "A detailed, personalized response from the chatbot."
}
```

#### Error Response (400/500):

```json
{
    "error": "Gemini failed to respond",
    "details": "..."
}
```

## Project Structure

```
.
├── app.py                      # Main Flask application file
├── food_consumption_model_xgb.pkl # Pre-trained machine learning model
├── requirements.txt            # Python dependencies
├── render.yaml                 # Deployment configuration for Render
├── .env                        # Local environment variables (for GENAI_API_KEY, not for git)
└── README.md                   # This file
```
