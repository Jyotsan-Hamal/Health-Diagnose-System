import pickle
import numpy as np
import pandas as pd

from tensorflow import keras


# Load the Keras model
model = keras.models.load_model('diagnose_system.h5')

data_df = pd.read_csv('./data/dataset.csv')
def total_unique_symptoms(df):
    #Returns all the uniques symptoms from all 17 columns
    lista =list( df['Symptom_1'].unique())
    for i in range(1,8):
        list_ =  list(df[f'Symptom_{i}'].unique())
        for j in list_:
            if j not in lista:
                lista.append(j)
            else:
                continue
    return lista

dis_list = ['Fungal infection',
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
 'Impetigo']

def check(user_input):
    lista = total_unique_symptoms(data_df)
    new_list = []
    for i in lista:
        if isinstance(i, str):  # Check if the element is a string
            new_list.append(i.strip())
        else:
            new_list.append(str(i))
            
    user_input_symptoms = user_input

    
    vec = np.zeros(len(new_list))
    for index in range(len(new_list)):
        for j in user_input_symptoms:
            if j == new_list[index]:
                
                vec[index]=1
        
    # Convert the list to a NumPy array and reshape it for model input
    user_input_array = np.array(vec).reshape(1, -1)
    # Make predictions with the trained model
    predictions = model.predict(user_input_array)

    # Get the predicted disease class (index with the highest probability)
    predicted_class = np.argmax(predictions)

    # Display the predicted class and its probability
    print(f"Predicted Disease Class: {predicted_class}")
    print(f"Probability Distribution: {predictions[0][predicted_class]}") 
    print(f"Your Disease is : {dis_list[predicted_class]}")

check(['chest_pain','breathlessness','sweating','vomiting'])