import streamlit as st
import numpy as np
import joblib

model=joblib.load("C:/Users/mital/OneDrive/Desktop/oasis internship/iris_model")

st.title("IRIS FLOWER PREDICTION")
st.write("ENTER DETAILS")

SepalLengthCm =st.text_input("SEPAL LENGHT (cm):")
SepalWidthCm  = st.text_input('SEPAL WIDTH (cm): ') 
PetalLengthCm =st.text_input('PETAL LENGHT (cm): ')
PetalWidthCm  =st.text_input("PETAL WIDTH (cm):")



def predict():
    try:
        # Convert input data to float
        row = [float(SepalLengthCm), float(SepalWidthCm), float(PetalLengthCm), float(PetalWidthCm)]
        input_data_as_numpy_array = np.asarray(row)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        
        # Make prediction
        predictions = model.predict(input_data_reshaped)
        
        # Display result
        if predictions == 0:
            result = "It is Iris-setosa"
        elif predictions == 1:
            result = "It is Iris-versicolor"
        else:
            result = "It is Iris-virginica"
        
        st.session_state['result'] = result
    except ValueError:
        st.session_state['result'] = "Please enter valid numeric values."

# Predict button
st.button("Predict", on_click=predict)

# Display the result at the bottom
if 'result' in st.session_state:
    st.write("## Prediction Result")
    st.write(st.session_state['result'])