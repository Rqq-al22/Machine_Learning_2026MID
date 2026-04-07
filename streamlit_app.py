import streamlit as st
import pandas as pd
import joblib

# ======================
# LOAD MODEL & KOMPONEN
# ======================
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca_model.pkl')
kmeans_model = joblib.load('kmeans_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# ======================
# TITLE
# ======================
st.title('Prediksi Klaster Gen Z Mental Wellness')
st.write('Aplikasi ini memprediksi klaster berdasarkan gaya hidup & kondisi mental.')

# ======================
# SIDEBAR INPUT
# ======================
st.sidebar.header('Input Data')

def user_input():
    data = {
        'Age': st.sidebar.slider('Age', 15, 30, 20),
        'Daily_Social_Media_Hours': st.sidebar.slider('Daily Social Media Hours', 0.0, 10.0, 3.0),
        'Screen_Time_Hours': st.sidebar.slider('Screen Time Hours', 0.0, 15.0, 7.0),
        'Night_Scrolling_Frequency': st.sidebar.slider('Night Scrolling Frequency', 0, 7, 3),
        'Online_Gaming_Hours': st.sidebar.slider('Online Gaming Hours', 0.0, 10.0, 2.0),
        'Exercise_Frequency_per_Week': st.sidebar.slider('Exercise Frequency per Week', 0, 7, 3),
        'Daily_Sleep_Hours': st.sidebar.slider('Daily Sleep Hours', 0.0, 12.0, 7.0),
        'Caffeine_Intake_Cups': st.sidebar.slider('Caffeine Intake Cups', 0, 5, 1),
        'Study_Work_Hours_per_Day': st.sidebar.slider('Study/Work Hours per Day', 0.0, 12.0, 6.0),
        'Overthinking_Score': st.sidebar.slider('Overthinking Score', 1, 10, 5),
        'Anxiety_Score': st.sidebar.slider('Anxiety Score', 1, 10, 5),
        'Mood_Stability_Score': st.sidebar.slider('Mood Stability Score', 1, 10, 5),
        'Social_Comparison_Index': st.sidebar.slider('Social Comparison Index', 1, 10, 5),
        'Sleep_Quality_Score': st.sidebar.slider('Sleep Quality Score', 1, 10, 5),
        'Motivation_Level': st.sidebar.slider('Motivation Level', 1, 10, 5),
        'Emotional_Fatigue_Score': st.sidebar.slider('Emotional Fatigue Score', 1, 10, 5),
        'Wellbeing_Index': st.sidebar.slider('Wellbeing Index', 1, 10, 5),

        # categorical
        'Gender': st.sidebar.selectbox('Gender', ['Male', 'Female']),
        'Country': st.sidebar.selectbox('Country', ['Indonesia', 'India', 'Other']),
        'Student_Working_Status': st.sidebar.selectbox('Status', ['Student', 'Working']),
        'Content_Type_Preference': st.sidebar.selectbox(
            'Content Preference',
            ['Music', 'Educational', 'Entertainment', 'News', 'Social Media', 'Sports']
        )
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input()

# ======================
# TAMPILKAN INPUT
# ======================
st.subheader('Data Input')
st.write(input_df)

# ======================
# PREDIKSI
# ======================
if st.button('Prediksi Klaster'):

    # ======================
    # ENCODING OTOMATIS
    # ======================
    processed = pd.get_dummies(input_df)

    # SAMAKAN DENGAN TRAINING
    processed = processed.reindex(columns=feature_columns, fill_value=0)

    # HANDLE NULL
    processed = processed.fillna(0)

    # ======================
    # SCALING
    # ======================
    scaled = scaler.transform(processed)

    # ======================
    # PCA
    # ======================
    pca_data = pca.transform(scaled)

    # ======================
    # PREDIKSI
    # ======================
    cluster = kmeans_model.predict(pca_data)[0]

    # ======================
    # OUTPUT
    # ======================
    st.subheader('Hasil Prediksi')
    st.success(f'Anda termasuk dalam Klaster: {cluster}')

    # ======================
    # INTERPRETASI (BONUS)
    # ======================
    if cluster == 0:
        st.info("💡 Klaster 0: Cenderung lebih stabil secara mental dan memiliki pola hidup seimbang.")
    elif cluster == 1:
        st.warning("⚠️ Klaster 1: Indikasi adanya tekanan mental atau pola hidup kurang sehat.")
