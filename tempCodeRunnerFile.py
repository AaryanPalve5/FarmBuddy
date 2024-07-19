from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import bz2
import pickle

app = Flask(__name__)

# Load the datasets for crop suitability
districts_df = pd.read_csv('maharashtra_districts_nutrients.csv')
crops_df = pd.read_csv('maharashtra_crops_nutrients.csv')

# Load the crop suitability model
model_path = 'crop_suitability_model.pkl'
if os.path.exists(model_path):
    crop_suitability_model = joblib.load(model_path)
else:
    crop_suitability_model = None
    print("Crop Suitability Model not found. Please ensure 'crop_suitability_model.pkl' exists.")

# Load the crop recommendation model
def decompress_pickle(file):
    with bz2.BZ2File(file, 'rb') as data:
        return pickle.load(data)

# Adjust the path to match your deployment environment
crop_recommendation_model = decompress_pickle('models/crop_recommendation_model.pbz2')

# Load dataset and fit label encoder for crop recommendation
df = pd.read_csv("Dataset/Crop_recommendation.csv", encoding='utf-8')
label_encoder = LabelEncoder()
label_encoder.fit(df['label'])

# Fertilizers dictionary for crop suitability
fertilizers = {
    'Nitrogen': 'Urea',
    'Phosphorus': 'Single Super Phosphate (SSP)',
    'Potassium': 'Muriate of Potash (MOP)'
}

# Function to suggest fertilizers based on nutrient gap for crop suitability
def suggest_fertilizers(nutrient_gap):
    suggestions = {}
    for nutrient, gap in nutrient_gap.items():
        if gap > 0:
            suggestions[nutrient] = {
                'Fertilizer': fertilizers[nutrient],
                'Amount': gap
            }
    return suggestions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/farmbuddy')
def farmbuddy():
    return render_template('farmbuddy.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/predict', methods=['POST'])
def predict():
    district = request.form.get('district').strip().lower()
    crop = request.form.get('crop').strip().lower()

    # Debug logging
    print(f"Received district: {district}, crop: {crop}")

    if crop_suitability_model:
        district_data = districts_df[districts_df['District'].str.lower() == district]
        crop_data = crops_df[crops_df['Crop'].str.lower() == crop]

        # Debug logging
        print(f"District data: {district_data}")
        print(f"Crop data: {crop_data}")

        if district_data.empty or crop_data.empty:
            result = "Invalid district or crop. Please try again."
        else:
            district_data = district_data.iloc[0]
            crop_data = crop_data.iloc[0]

            input_data = pd.DataFrame([[
                district_data['Nitrogen'],
                district_data['Phosphorus'],
                district_data['Potassium']
            ]], columns=['Nitrogen_district', 'Phosphorus_district', 'Potassium_district'])

            is_suitable = crop_suitability_model.predict(input_data)[0]

            if is_suitable:
                result = f"The crop '{crop}' is suitable for the district '{district}'."
            else:
                nutrient_gap = {
                    'Nitrogen': crop_data['Nitrogen'] - district_data['Nitrogen'],
                    'Phosphorus': crop_data['Phosphorus'] - district_data['Phosphorus'],
                    'Potassium': crop_data['Potassium'] - district_data['Potassium']
                }

                threshold = 20
                if any(gap > threshold for gap in nutrient_gap.values()):
                    result = f"The crop '{crop}' is not suitable for the district '{district}' due to high nutrient deficiencies."
                else:
                    suggestions = suggest_fertilizers(nutrient_gap)
                    result = f"The crop '{crop}' is not suitable for the district '{district}' due to nutrient deficiencies. Suggested fertilizers: {suggestions}"
    else:
        result = "Crop Suitability Model not loaded."

    return render_template('result.html', result=result)

@app.route('/crop_home')
def crop_home():
    return render_template('crop_home.html')

@app.route('/crop_index')
def crop_index():
    return render_template('crop_index.html')

@app.route('/crop_parameters', methods=['POST'])
def crop_parameters():
    try:
        # Retrieve form data
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        
        # Make prediction using crop recommendation model
        predicted_crop = crop_recommendation_model.predict([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # Decode the prediction
        decoded_labels = label_encoder.inverse_transform(predicted_crop)
        predicted_crop = decoded_labels[0]
        
        # Render the result template with prediction results
        return render_template('crop_result.html', crop=predicted_crop)
    
    except Exception as e:
        # Log and print any errors
        print(f"Error: {e}")
        return render_template('crop_result.html', crop="Error occurred during prediction")

if __name__ == "__main__":
    app.run(debug=True)
