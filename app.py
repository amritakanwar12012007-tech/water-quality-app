import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Dataset
data = {
    'pH':[7.2,6.8,8.1,7.5,6.9,7.0,8.0,6.7,7.8,7.1],
    'TDS':[300,500,200,400,350,320,210,480,260,330],
    'DO':[6,5,7,6.5,5.5,6.2,7.1,5.2,6.8,6.0],
    'BOD':[2,3,1,2.5,2.8,2.2,1.2,3.1,1.8,2.4],
    'WQI':[45,60,30,50,55,48,28,65,35,52]
}

df = pd.DataFrame(data)

X = df[['pH','TDS','DO','BOD']]
y = df['WQI']

model = RandomForestRegressor()
model.fit(X,y)

st.title("💧 Water Quality Prediction System")
st.write("Enter water parameters to predict Water Quality Index")

pH = st.number_input("Enter pH")
TDS = st.number_input("Enter TDS")
DO = st.number_input("Enter DO")
BOD = st.number_input("Enter BOD")

if st.button("Predict"):
    new_data = pd.DataFrame([[pH,TDS,DO,BOD]], columns=['pH','TDS','DO','BOD'])
    prediction = model.predict(new_data)[0]

    st.write("Predicted WQI:", prediction)

    if prediction <= 25:
        st.success("Excellent Water Quality")
    elif prediction <= 50:
        st.success("Good Water Quality")
    elif prediction <= 75:
        st.warning("Poor Water Quality")
    else:
        st.error("Very Poor Water Quality")
