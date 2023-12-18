import pickle
import numpy as np
import pandas as pd

from tensorflow import keras


# Load the Keras model
model = keras.models.load_model('diagnose_system.h5')

class Health():
    def __init__(self) -> None:
        self.symptoms = ['itching',
                        'skin_rash',
                        'continuous_sneezing',
                        'shivering',
                        'stomach_pain',
                        'acidity',
                        'vomiting',
                        'indigestion',
                        'muscle_wasting',
                        'patches_in_throat',
                        'fatigue',
                        'weight_loss',
                        'sunken_eyes',
                        'cough',
                        'headache',
                        'chest_pain',
                        'back_pain',
                        'weakness_in_limbs',
                        'chills',
                        'joint_pain',
                        'yellowish_skin',
                        'constipation',
                        'pain_during_bowel_movements',
                        'breathlessness',
                        'cramps',
                        'weight_gain',
                        'mood_swings',
                        'neck_pain',
                        'muscle_weakness',
                        'stiff_neck',
                        'pus_filled_pimples',
                        'burning_micturition',
                        'bladder_discomfort',
                        'high_fever',
                        'nodal_skin_eruptions',
                        'ulcers_on_tongue',
                        'loss_of_appetite',
                        'restlessness',
                        'dehydration',
                        'dizziness',
                        'weakness_of_one_body_side',
                        'lethargy',
                        'nausea',
                        'abdominal_pain',
                        'pain_in_anal_region',
                        'sweating',
                        'bruising',
                        'cold_hands_and_feets',
                        'anxiety',
                        'knee_pain',
                        'swelling_joints',
                        'blackheads',
                        'foul_smell_of urine',
                        'skin_peeling',
                        'blister',
                        'dischromic _patches',
                        'watering_from_eyes',
                        'extra_marital_contacts',
                        'diarrhoea',
                        'loss_of_balance',
                        'blurred_and_distorted_vision',
                        'altered_sensorium',
                        'dark_urine',
                        'swelling_of_stomach',
                        'bloody_stool',
                        'obesity',
                        'hip_joint_pain',
                        'movement_stiffness',
                        'spinning_movements',
                        'scurring',
                        'continuous_feel_of_urine',
                        'silver_like_dusting',
                        'red_sore_around_nose',
                        'nan',
                        'spotting_ urination',
                        'passage_of_gases',
                        'irregular_sugar_level',
                        'family_history',
                        'lack_of_concentration',
                        'excessive_hunger',
                        'yellowing_of_eyes',
                        'distention_of_abdomen',
                        'irritation_in_anus',
                        'swollen_legs',
                        'painful_walking',
                        'small_dents_in_nails',
                        'yellow_crust_ooze',
                        'internal_itching',
                        'mucoid_sputum',
                        'history_of_alcohol_consumption',
                        'swollen_blood_vessels',
                        'unsteadiness',
                        'inflammatory_nails',
                        'depression',
                        'fluid_overload',
                        'swelled_lymph_nodes',
                        'malaise',
                        'prominent_veins_on_calf',
                        'puffy_face_and_eyes',
                        'fast_heart_rate',
                        'irritability',
                        'muscle_pain',
                        'mild_fever',
                        'yellow_urine',
                        'phlegm',
                        'enlarged_thyroid']
                                
        self.dis_list = ['Fungal infection',
            'Allergy',
            'GERD',
            'Chronic cholestasis',
            'Drug Reaction',
            'Peptic ulcer diseae',
            'AIDS',
            'Diabetes ',
            'Gastroenteritis',
            'Bronchial Asthma',
            'Hypertension ',
            'Migraine',
            'Cervical spondylosis',
            'Paralysis (brain hemorrhage)',
            'Jaundice',
            'Malaria',
            'Chicken pox',
            'Dengue',
            'Typhoid',
            'hepatitis A',
            'Hepatitis B',
            'Hepatitis C',
            'Hepatitis D',
            'Hepatitis E',
            'Alcoholic hepatitis',
            'Tuberculosis',
            'Common Cold',
            'Pneumonia',
            'Dimorphic hemmorhoids(piles)',
            'Heart attack',
            'Varicose veins',
            'Hypothyroidism',
            'Hyperthyroidism',
            'Hypoglycemia',
            'Osteoarthristis',
            'Arthritis',
            '(vertigo) Paroymsal  Positional Vertigo',
            'Acne',
            'Urinary tract infection',
            'Psoriasis',
            'Impetigo'
            ]
    def check(self,user_input):
           
        user_input_symptoms = user_input

        
        vec = np.zeros(len(self.symptoms))
        for index in range(len(self.symptoms)):
            for j in user_input_symptoms:
                if j == self.symptoms[index]:
                    
                    vec[index]=1
            
        # Convert the list to a NumPy array and reshape it for model input
        user_input_array = np.array(vec).reshape(1, -1)
        # Make predictions with the trained model
        predictions = model.predict(user_input_array)

                # Get the indices of the top 5 predicted classes
        top_classes_indices = np.argsort(predictions)[0, -5:][::-1]

        # Display the top 5 predicted classes and their probabilities
        print("Top 5 Predicted Disease Classes:")
        for index in top_classes_indices:
            disease_class = self.dis_list[index]
            probability = predictions[0][index]
            print(f"{disease_class}: {probability}")

        # Return the top predicted disease class
        return self.dis_list[top_classes_indices[0]]



#blurred_and_distorted_vision




def main():
    obj = Health()
    obj.check(['history_of_alcohol_consumption','swollen_legs','family_history','obesity','bloody_stool','blurred_and_distorted_vision','anxiety','weight_gain','chest_pain','headache',''])
    
    
if __name__=='__main__':
    main()