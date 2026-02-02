from flask import Flask, render_template, request
import xgboost as xgb
import numpy as np

app = Flask(__name__)

# Load model from JSON
model = xgb.Booster()
model.load_model("xgb_model.json")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        fields = [
            "Age", "Number_of_sexual_partners", "First_sexual_intercourse", "Num_of_pregnancies",
            "Smokes", "Smokes_years", "Smokes_packs_year", "Hormonal_Contraceptives",
            "Hormonal_Contraceptives_years", "IUD", "IUD_years", "STDs", "STDs_number",
            "STDs_condylomatosis", "STDs_cervical_condylomatosis", "STDs_vaginal_condylomatosis",
            "STDs_vulvo_perineal_condylomatosis", "STDs_syphilis", "STDs_pelvic_inflammatory_disease",
            "STDs_genital_herpes", "STDs_molluscum_contagiosum", "STDs_AIDS", "STDs_HIV",
            "STDs_Hepatitis_B", "STDs_HPV", "STDs_Number_of_diagnosis", "STDs_Time_since_first_diagnosis",
            "STDs_Time_since_last_diagnosis", "Dx_Cancer", "Dx_CIN", "Dx_HPV", "Biopsy"
        ]

        # Convert inputs to float and to DMatrix
        input_features = [float(request.form[field]) for field in fields]
        input_array = np.array([input_features])
        dmatrix = xgb.DMatrix(input_array)

        # Predict probability
        prediction_proba = model.predict(dmatrix)[0]  # probability of class 1

        # Format prediction as percentage
        return render_template('index.html', prediction_text=f'Cervical Cancer Risk Probability: {prediction_proba:.2%}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
