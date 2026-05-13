import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Dataset
data = {
    'pH':[7.2,6.8,8.1,7.5,6.9,7.0,8.0,6.7,7.8,7.1],
    'TDS':[300,500,200,400,350,320,210,480,260,330],
    'DO':[6,5,7,6.5,5.5,6.2,7.1,5.2,6.8,6.0],
    'BOD':[2,3,1,2.5,2.8,2.2,1.2,3.1,1.8,2.4],
    'WQI':[45,60,30,50,55,48,28,65,35,52]
}

df = pd.DataFrame(data)

# Model data
X = df[['pH','TDS','DO','BOD']]
y = df['WQI']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
rf = RandomForestRegressor()
lr = LinearRegression()
dt = DecisionTreeRegressor()

# Train
rf.fit(X_train, y_train)
lr.fit(X_train, y_train)
dt.fit(X_train, y_train)

# Accuracy
rf_score = r2_score(y_test, rf.predict(X_test))
lr_score = r2_score(y_test, lr.predict(X_test))
dt_score = r2_score(y_test, dt.predict(X_test))

# Title
st.title("💧 Water Quality Prediction System")
st.caption("AI-based Water Quality Monitoring System")
st.write("Enter water parameters to predict Water Quality Index")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Graphs", "📈 Model Info"])

# ---------------- TAB 1 ----------------
with tab1:
    pH = st.number_input("Enter pH", 0.0, 14.0)
    TDS = st.number_input("Enter TDS", 0, 1000)
    DO = st.number_input("Enter DO", 0.0, 14.0)
    BOD = st.number_input("Enter BOD", 0.0, 10.0)

    if st.button("Predict"):
        new_data = pd.DataFrame([[pH, TDS, DO, BOD]], columns=['pH','TDS','DO','BOD'])
        prediction = rf.predict(new_data)[0]

        st.subheader("Prediction Result")
        st.write("Predicted WQI:", round(prediction,2))

        if prediction < 50:
            st.success("🟢 Good Water Quality - Safe for Drinking")
        elif prediction < 75:
            st.warning("🟡 Moderate - Needs Treatment")
        else:
            st.error("🔴 Poor - Not Safe for Drinking")

        st.info(f"Prediction Confidence: {round(rf_score*100,2)}%")

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("TDS vs WQI Graph")

    fig, ax = plt.subplots()
    ax.scatter(df['TDS'], df['WQI'])
    ax.set_xlabel("TDS")
    ax.set_ylabel("WQI")
    st.pyplot(fig)

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("📊 Model Comparison")

    score_df = pd.DataFrame({
        "Model": ["Random Forest", "Linear Regression", "Decision Tree"],
        "R² Score": [rf_score, lr_score, dt_score]
    })

    st.bar_chart(score_df.set_index("Model"))

    st.subheader("📌 Feature Importance")

    importance = rf.feature_importances_
    features = ['pH','TDS','DO','BOD']

    imp_df = pd.DataFrame({'Feature':features,'Importance':importance})
    st.bar_chart(imp_df.set_index('Feature'))

# Footer
st.write("Developed for Water Quality Analysis Project")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}
h1, h2, h3 {
    color: #E0F7FA;
}
</style>
""", unsafe_allow_html=True)

st.image("https://images.unsplash.com/photo-1507525428034-b723cf961d3e", use_container_width=True)
