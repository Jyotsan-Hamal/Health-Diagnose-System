import pandas as pd
import numpy as np
from tensorflow import keras




data_df = pd.read_csv("./data/dataset.csv")

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


def total_unique_symptoms(data_df):
    # Returns all the uniques symptoms from all 17 columns
    
    disease_list =list( data_df['Symptom_1'].unique())
    # Creates a unique list of disease from first column from our dataset
    for i in range(2,8):
        # Create a Unique list of disease from symptom 1 to 8 columns
        temp_list =  list(data_df[f'Symptom_{i}'].unique())
        # iterate each items inside that list
        for j in temp_list:
            # Check if that disease is already in our list 
            if j not in disease_list:
                # If not then append the diesease and move to next column.
                disease_list.append(j)
            
    return disease_list
     
     
    
  
def get_dis_vectors():   
    # return the 43 vectors for the disease 
    
    dis_vectors = []
    for i in range(len(data_df)):
        vec = np.zeros(len(dis_list))
        for j in range(len(dis_list)):
            if data_df.iloc[i][0] == dis_list[j]:
                vec[j] = 1
        dis_vectors.append(vec)
    return dis_vectors
             
    
      

def Data_Algorithm():
    #Returns the All unique symptoms into a  103 vectors
    lista = total_unique_symptoms(data_df)
    
    Disease = []
    Vectors = []
    for i in range(len(data_df)):
        vec = np.zeros(len(lista))
        pre = data_df.iloc[i].tolist()
        Disease.append(pre[0])
        for dis in range(len(lista)):
            for j in pre[1:]:
                if j == lista[dis]:
                    vec[dis] = 1
        Vectors.append(vec)
    
    
    
    has = {
        'Disease':Disease,
        'Vectors':Vectors
    }
    df = pd.DataFrame(has)
    return df



df = Data_Algorithm()    
        
from sklearn.model_selection import train_test_split
# Convert the 'Symptoms' column to a NumPy array
symptoms_array = np.vstack(df['Vectors'].tolist())#<-----This one is important remember!!!!!!!!!!!!

dis = get_dis_vectors() #<--- this function provides all the vectord for 43 disease.

labels = np.vstack(dis) #<-- converts into numpy array format for training the format.


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(symptoms_array, labels, test_size=0.2, random_state=42)





# Define the ANN model
model = keras.Sequential([
    keras.layers.Input(shape=(106,)),  # Input layer with 103 neurons
    keras.layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons and ReLU activation
    keras.layers.Dense(128, activation='relu'), 
    keras.layers.Dense(41, activation='softmax')  # Output layer with 41 neurons (one for each disease) and softmax activation for multi-class classification
])

# Compile the model
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
    )

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


def check(user_input):
    lista = total_unique_symptoms(data_df)
    new_list = []
    for i in lista:
        if isinstance(i, str):  # Check if the element is a string
            new_list.append(i.strip())
        else:
            new_list.append(i)
            
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
    

check(['vomiting','breathlessness','sweating','chest_pain'])
