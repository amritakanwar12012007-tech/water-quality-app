import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# ---------------- UI ----------------
st.title("💧 AquaAI - Water Quality Monitoring System")
st.caption("AI-based Smart Water Quality Prediction")

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Graphs", "📈 Model Info"])

# ---------------- TAB 1 ----------------
with tab1:

    # CSV Upload
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset uploaded successfully")
    else:
        # Default dataset
        data = {
            'pH':[7.2,6.8,8.1,7.5,6.9,7.0,8.0,6.7,7.8,7.1],
            'TDS':[300,500,200,400,350,320,210,480,260,330],
            'DO':[6,5,7,6.5,5.5,6.2,7.1,5.2,6.8,6.0],
            'BOD':[2,3,1,2.5,2.8,2.2,1.2,3.1,1.8,2.4],
            'WQI':[45,60,30,50,55,48,28,65,35,52]
        }
        df = pd.DataFrame(data)

    # Model training
    X = df[['pH','TDS','DO','BOD']]
    y = df['WQI']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rf = RandomForestRegressor()
    lr = LinearRegression()
    dt = DecisionTreeRegressor()

    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    dt.fit(X_train, y_train)

    rf_score = r2_score(y_test, rf.predict(X_test))
    lr_score = r2_score(y_test, lr.predict(X_test))
    dt_score = r2_score(y_test, dt.predict(X_test))

    # Auto-fill
    if st.button("Use Sample Data"):
        st.session_state.pH = 7.0
        st.session_state.TDS = 300
        st.session_state.DO = 6.0
        st.session_state.BOD = 2.5

    pH = st.number_input("Enter pH", 0.0, 14.0, value=st.session_state.get("pH", 7.0))
    TDS = st.number_input("Enter TDS", 0, 1000, value=st.session_state.get("TDS", 300))
    DO = st.number_input("Enter DO", 0.0, 14.0, value=st.session_state.get("DO", 6.0))
    BOD = st.number_input("Enter BOD", 0.0, 10.0, value=st.session_state.get("BOD", 2.5))

    # Prediction
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

        # Warnings
        if BOD > 5:
            st.error("⚠️ High pollution detected")
        if DO < 4:
            st.warning("⚠️ Low oxygen level")

        st.info(f"Prediction Confidence: {round(rf_score*100,2)}%")

        # Save history
        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "pH": pH,
            "TDS": TDS,
            "DO": DO,
            "BOD": BOD,
            "WQI": round(prediction,2)
        })

        # PDF Download
        if st.button("Download Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            pdf.cell(200, 10, txt="Water Quality Report", ln=True)
            pdf.cell(200, 10, txt=f"pH: {pH}", ln=True)
            pdf.cell(200, 10, txt=f"TDS: {TDS}", ln=True)
            pdf.cell(200, 10, txt=f"DO: {DO}", ln=True)
            pdf.cell(200, 10, txt=f"BOD: {BOD}", ln=True)
            pdf.cell(200, 10, txt=f"WQI: {round(prediction,2)}", ln=True)

            pdf.output("report.pdf")

            with open("report.pdf", "rb") as f:
                st.download_button("Download PDF", f, file_name="Water_Report.pdf")

# ---------------- TAB 2 ----------------
with tab2:

    st.subheader("TDS vs WQI Graph")
    fig, ax = plt.subplots()
    ax.scatter(df['TDS'], df['WQI'])
    ax.set_xlabel("TDS")
    ax.set_ylabel("WQI")
    st.pyplot(fig)

    st.subheader("pH vs WQI Graph")
    fig2, ax2 = plt.subplots()
    ax2.scatter(df['pH'], df['WQI'])
    ax2.set_xlabel("pH")
    ax2.set_ylabel("WQI")
    st.pyplot(fig2)

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

    # History
    if "history" in st.session_state:
        st.subheader("📜 Prediction History")
        st.write(pd.DataFrame(st.session_state.history))

# ---------------- FOOTER ----------------
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
