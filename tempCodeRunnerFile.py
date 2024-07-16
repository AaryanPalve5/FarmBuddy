from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the datasets
districts_df = pd.read_csv('maharashtra_districts_nutrients.csv')
crops_df = pd.read_csv('maharashtra_crops_nutrients.csv')

# Load the model
model_path = 'crop_suitability_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None
    print("Model not found. Please ensure 'crop_suitability_model.pkl' exists.")

fertilizers = {
    'Nitrogen': 'Urea',
    'Phosphorus': 'Single Super Phosphate (SSP)',
    'Potassium': 'Muriate of Potash (MOP)'
}

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

@app.route('/predict', methods=['POST'])
def predict():
    district = request.form.get('district').strip().lower()
    crop = request.form.get('crop').strip().lower()

    if model:
        district_data = districts_df[districts_df['District'].str.lower() == district]
        crop_data = crops_df[crops_df['Crop'].str.lower() == crop]

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

            is_suitable = model.predict(input_data)[0]

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
        result = "Model not loaded."

    return render_template('result.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
