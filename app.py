import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---------------- CUSTOM STYLE ----------------
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: Georgia, serif;
}

.stApp {
    background-color: #f5f7fa;
    color: #333333;
}

/* Buttons */
.stButton>button {
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
    border: none;
}

/* Predict button */
div.stButton:nth-child(6) button {
    background-color: #28a745;
    color: white;
}

/* Sample button */
div.stButton:nth-child(5) button {
    background-color: #007bff;
    color: white;
}

h1, h2, h3 {
    color: #2c3e50;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
col1, col2 = st.columns([1, 5])

with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/728/728093.png", width=60)

with col2:
    st.title("AquaAI")
    st.markdown("### Clean Water • Smart Decisions")

st.write("---")

# ---------------- DATA ----------------
data = {
    'pH':[7.2,6.8,8.1,7.5,6.9,7.0,8.0,6.7,7.8,7.1],
    'TDS':[300,500,200,400,350,320,210,480,260,330],
    'DO':[6,5,7,6.5,5.5,6.2,7.1,5.2,6.8,6.0],
    'BOD':[2,3,1,2.5,2.8,2.2,1.2,3.1,1.8,2.4],
    'WQI':[45,60,30,50,55,48,28,65,35,52]
}

df = pd.DataFrame(data)

# ---------------- MODEL ----------------
X = df[['pH','TDS','DO','BOD']]
y = df['WQI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Insights"])

# ---------------- TAB 1 ----------------
with tab1:

    st.subheader("Enter Water Parameters")

    if st.button("Use Sample Data"):
        st.session_state.pH = 7.0
        st.session_state.TDS = 300
        st.session_state.DO = 6.0
        st.session_state.BOD = 2.5

    pH = st.number_input("pH", 0.0, 14.0, value=st.session_state.get("pH", 7.0))
    TDS = st.number_input("TDS", 0, 1000, value=st.session_state.get("TDS", 300))
    DO = st.number_input("DO", 0.0, 14.0, value=st.session_state.get("DO", 6.0))
    BOD = st.number_input("BOD", 0.0, 10.0, value=st.session_state.get("BOD", 2.5))

    if st.button("Predict"):
        new_data = pd.DataFrame([[pH, TDS, DO, BOD]], columns=['pH','TDS','DO','BOD'])
        prediction = rf.predict(new_data)[0]

        st.subheader("Result")
        st.write(f"WQI: {round(prediction,2)}")

        if prediction < 50:
            st.success("🟢 Good Water Quality")
        elif prediction < 75:
            st.warning("🟡 Moderate Quality")
        else:
            st.error("🔴 Poor Quality")

        if BOD > 5:
            st.error("⚠️ High pollution detected")
        if DO < 4:
            st.warning("⚠️ Low oxygen level")

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

    st.subheader("Water Quality Distribution")

    categories = pd.cut(df['WQI'],
                        bins=[0,50,75,100],
                        labels=["Good","Moderate","Poor"])

    counts = categories.value_counts()

    fig, ax = plt.subplots()
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%')
    ax.set_title("Water Quality Levels")

    st.pyplot(fig)

# ---------------- FOOTER ----------------
st.write("---")
st.write("Developed for Water Quality Analysis Project")
