import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Dataset
data = {
    'pH':[7.2,6.8,8.1,7.5,6.9,7.0,8.0,6.7,7.8,7.1],
    'TDS':[300,500,200,400,350,320,210,480,260,330],
    'DO':[6,5,7,6.5,5.5,6.2,7.1,5.2,6.8,6.0],
    'BOD':[2,3,1,2.5,2.8,2.2,1.2,3.1,1.8,2.4],
    'WQI':[45,60,30,50,55,48,28,65,35,52]
}

df = pd.DataFrame(data)

# Model
X = df[['pH','TDS','DO','BOD']]
y = df['WQI']

model = RandomForestRegressor()
model.fit(X,y)

# Title
st.title("💧 Water Quality Prediction System")
st.write("Enter water parameters to predict Water Quality Index")

# Inputs
pH = st.number_input("Enter pH", 0.0, 14.0)
TDS = st.number_input("Enter TDS", 0, 1000)
DO = st.number_input("Enter DO", 0.0, 14.0)
BOD = st.number_input("Enter BOD", 0.0, 10.0)

# Prediction
if st.button("Predict"):
    new_data = pd.DataFrame([[pH,TDS,DO,BOD]], columns=['pH','TDS','DO','BOD'])
    prediction = model.predict(new_data)[0]

    st.subheader("Prediction Result")
    st.write("Predicted WQI:", round(prediction,2))

    # Category
    if prediction <= 25:
        st.success("Excellent Water Quality")
    elif prediction <= 50:
        st.success("Good Water Quality")
    elif prediction <= 75:
        st.warning("Poor Water Quality")
    else:
        st.error("Very Poor Water Quality")

    # Recommendation
    if prediction > 50:
        st.warning("Suggestion: Treat water before use")
    else:
        st.success("Water is safe for use")

# Graph
st.subheader("TDS vs WQI Graph")
fig, ax = plt.subplots()
ax.scatter(df['TDS'], df['WQI'])
ax.set_xlabel("TDS")
ax.set_ylabel("WQI")
st.pyplot(fig)

# Feature Importance
importance = model.feature_importances_
features = ['pH','TDS','DO','BOD']

imp_df = pd.DataFrame({'Feature':features,'Importance':importance})

st.subheader("Feature Importance")
st.bar_chart(imp_df.set_index('Feature'))

# Footer
st.write("Developed for Water Quality Analysis Project")
