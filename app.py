from flask import Flask, render_template, request
from works import Health
import pandas as pd
app = Flask(__name__)

symptoms_list = [
                 'itching', 'skin_rash', 'continuous_sneezing', 'shivering', 'stomach_pain', 'acidity', 'vomiting',
                 'indigestion', 'muscle_wasting', 'patches_in_throat', 'fatigue', 'weight_loss', 'sunken_eyes', 'cough',
                 'headache', 'chest_pain', 'back_pain', 'weakness_in_limbs', 'chills', 'joint_pain', 'yellowish_skin',
                 'constipation', 'pain_during_bowel_movements', 'breathlessness', 'cramps', 'weight_gain', 'mood_swings',
                 'neck_pain', 'muscle_weakness', 'stiff_neck', 'pus_filled_pimples', 'burning_micturition',
                 'bladder_discomfort', 'high_fever', 'nodal_skin_eruptions', 'ulcers_on_tongue', 'loss_of_appetite',
                 'restlessness', 'dehydration', 'dizziness', 'weakness_of_one_body_side', 'lethargy', 'nausea',
                 'abdominal_pain', 'pain_in_anal_region', 'sweating', 'bruising', 'cold_hands_and_feets', 'anxiety',
                 'knee_pain', 'swelling_joints', 'blackheads', 'foul_smell_of urine', 'skin_peeling', 'blister',
                 'dischromic _patches', 'watering_from_eyes', 'extra_marital_contacts', 'diarrhoea', 'loss_of_balance',
                 'blurred_and_distorted_vision', 'altered_sensorium', 'dark_urine', 'swelling_of_stomach', 'bloody_stool',
                 'obesity', 'hip_joint_pain', 'movement_stiffness', 'spinning_movements', 'scurring',
                 'continuous_feel_of_urine', 'silver_like_dusting', 'red_sore_around_nose', 'nan', 'spotting_ urination',
                 'passage_of_gases', 'irregular_sugar_level', 'family_history', 'lack_of_concentration',
                 'excessive_hunger', 'yellowing_of_eyes', 'distention_of_abdomen', 'irritation_in_anus', 'swollen_legs',
                 'painful_walking', 'small_dents_in_nails', 'yellow_crust_ooze', 'internal_itching', 'mucoid_sputum',
                 'history_of_alcohol_consumption', 'swollen_blood_vessels', 'unsteadiness', 'inflammatory_nails',
                 'depression', 'fluid_overload', 'swelled_lymph_nodes', 'malaise', 'prominent_veins_on_calf',
                 'puffy_face_and_eyes', 'fast_heart_rate', 'irritability', 'muscle_pain', 'mild_fever', 'yellow_urine',
                 'phlegm', 'enlarged_thyroid'
                ]

obj = Health()


@app.route('/')
def index():
    return render_template('index.html', symptoms=symptoms_list)

@app.route('/process_symptoms', methods=['POST'])
def process_symptoms():
    selected_symptoms = request.form.getlist('symptoms')
    diseases = obj.check(selected_symptoms)
    des = pd.read_csv("./data/symptom_Description.csv")
    pre = pd.read_csv("./data/symptom_precaution.csv")
    Description = des[des['Disease']==diseases]
    Description = Description.iloc[0,1]
    
    prec = pre[pre['Disease']==diseases]
    preco = prec.iloc[0,:].tolist()
    # precaution = des[des['Disease']==diseases]['precaution']
    haso = {
        "Selected_symptoms": selected_symptoms,
        'Dis':diseases,
        'Description':Description,
        'precaution':preco[1:]
        
        
        
    }
    return render_template('final.html', diseases=haso)

if __name__ == "__main__":
    app.run(debug=True)

