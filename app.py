from flask import Flask, request, jsonify, render_template
import joblib
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

# Configure Gemini AI
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6LFpThGhzpMwKIBDOzjI_Vu_lvNw47OfkgCc91_VmUyvA"))
llm_model = genai.GenerativeModel('gemini-2.5-flash')

app = Flask(__name__)

# Load local ML components
ml_model = joblib.load('agri_robot_model.pkl')
le_plant = joblib.load('plant_encoder.pkl')
le_disease = joblib.load('disease_encoder.pkl')

@app.route('/')
def home():
    plants = sorted(list(le_plant.classes_))
    diseases = sorted(list(le_disease.classes_))
    return render_template('index.html', plants=plants, diseases=diseases)

# NEW ROUTE: Translates the dropdown lists dynamically via AI
@app.route('/translate_ui', methods=['POST'])
def translate_ui():
    data = request.json
    language = data.get('language', 'English')
    
    plants = sorted(list(le_plant.classes_))
    diseases = sorted(list(le_disease.classes_))

    if language == 'English':
        return jsonify({"plants": plants, "diseases": diseases})

    try:
        prompt = f"""
        Translate the following agricultural terms into {language}.
        Return ONLY a raw JSON object with two keys: "plants" (an array of translated plant names) and "diseases" (an array of translated disease names). Do not include markdown formatting or backticks.
        Plants: {plants}
        Diseases: {diseases}
        """
        response = llm_model.generate_content(prompt)
        
        # Clean the response to ensure valid JSON parsing
        cleaned_text = response.text.strip().replace('```json', '').replace('```', '')
        translated_data = json.loads(cleaned_text)
        return jsonify(translated_data)
        
    except Exception as e:
        return jsonify({"error": "Translation failed", "plants": plants, "diseases": diseases})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    plant_input = data.get('plant', '').strip()
    disease_input = data.get('disease', '').strip()
    language = data.get('language', 'English').strip()

    try:
        # STEP 1: Translate the user's input BACK to English for the ML Model
        english_plant = plant_input
        english_disease = disease_input

        if language != 'English':
            translation_prompt = f"""
            Translate these agricultural terms from {language} to English.
            Return ONLY a raw JSON object with keys "plant" and "disease". Do not use markdown backticks.
            Plant: "{plant_input}"
            Disease: "{disease_input}"
            """
            trans_response = llm_model.generate_content(translation_prompt)
            cleaned_trans = trans_response.text.strip().replace('```json', '').replace('```', '')
            eng_data = json.loads(cleaned_trans)
            
            english_plant = eng_data.get("plant", plant_input)
            english_disease = eng_data.get("disease", disease_input)

        known_plants = list(le_plant.classes_)
        known_diseases = list(le_disease.classes_)

        # SCENARIO A: Exact Match Found in Local Database (Using translated English terms)
        if english_plant in known_plants and english_disease in known_diseases:
            plant_num = le_plant.transform([english_plant])[0]
            disease_num = le_disease.transform([english_disease])[0]
            ml_prediction = ml_model.predict([[plant_num, disease_num]])[0]

            prompt = f"""
            You are an expert agricultural botanist. Respond entirely in {language}.
            A farmer has a {english_plant} crop suffering from {english_disease}. 
            Our local database suggests this treatment: {ml_prediction}.
            
            Please provide:
            1. A translation of the plant name '{english_plant}', disease '{english_disease}', and treatment '{ml_prediction}' into {language}.
            2. A concise report with these sections:
               * **Root Causes:** How it develops.
               * **Best Herbal Practices:** Organic application.
               * **Long-Term Prevention:** Non-chemical methods.
            Format in Markdown. Ensure the tone is helpful for a farmer.
            """
            response = llm_model.generate_content(prompt)

            return jsonify({
                'status': f'✅ Local Match Analyzed',
                'treatment': f'{language} Diagnosis Generated',
                'report': response.text
            })

        # SCENARIO B: Unknown Plant or Disease
        else:
            prompt = f"""
            You are an expert agricultural botanist. Respond entirely in {language}.
            A farmer has inputted a plant: '{english_plant}' with symptoms: '{english_disease}'. 
            This is not in our local database.
            
            Please provide a comprehensive diagnostic report in {language} including:
            * **Possible Diseases:** What are the most likely issues?
            * **Recommended Treatment:** Best organic/herbal treatments.
            * **Long-Term Prevention:** Future safety.
            Format in Markdown.
            """
            response = llm_model.generate_content(prompt)

            return jsonify({
                'status': f'🌐 Cloud AI Deep Analysis',
                'treatment': f'{language} Diagnostic Complete',
                'report': response.text
            })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
