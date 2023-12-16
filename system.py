import pandas as pd
import numpy as np


data_df = pd.read_csv("./data/dataset.csv")

def total_unique_symptoms(df):
    lista =list( df['Symptom_1'].unique())
    for i in range(1,8):
        list_ =  list(df[f'Symptom_{i}'].unique())
        for j in list_:
            if j not in lista:
                lista.append(j)
            else:
                continue
    return lista
     
     
     
    
      

def Data_Algorithm():
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


# Convert the 'Disease' column to one-hot encoded labels
labels = pd.get_dummies(df['Disease']).values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(symptoms_array, labels, test_size=0.2, random_state=42)



from tensorflow import keras

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
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)


def check(user_input):
    lista = total_unique_symptoms(df)
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
