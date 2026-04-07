import streamlit as st
import pandas as pd
import joblib

# Load the saved models and preprocessors
# Ensure these files (scaler.pkl, pca_model.pkl, kmeans_model.pkl) are in the same directory as your Streamlit app
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca_model.pkl')
kmeans_model = joblib.load('kmeans_model.pkl')

# Define the numerical columns used for scaling and PCA
# This list should match the `numerical_cols` used during training
# Based on the notebook, these are:
numerical_cols_for_app = [
    'Age', 'Daily_Social_Media_Hours', 'Screen_Time_Hours',
    'Night_Scrolling_Frequency', 'Online_Gaming_Hours', 'Exercise_Frequency_per_Week',
    'Daily_Sleep_Hours', 'Caffeine_Intake_Cups', 'Study_Work_Hours_per_Day',
    'Overthinking_Score', 'Anxiety_Score', 'Mood_Stability_Score',
    'Social_Comparison_Index', 'Sleep_Quality_Score', 'Motivation_Level',
    'Emotional_Fatigue_Score', 'Wellbeing_Index'
]

# Define the categorical columns and their possible values (for one-hot encoding)
# This needs to be consistent with the one-hot encoding during training
categorical_cols_info = {
    'Gender': ['Female', 'Male'], # Assuming 'Other' was dropped (drop_first=True) or not present
    'Country': ['India', 'Other'], # Assuming 'Indonesia' was the first and dropped
    'Student_Working_Status': ['Working'], # Assuming 'Student' was the first and dropped
    'Content_Type_Preference': ['Educational', 'Entertainment', 'News', 'Social Media', 'Sports'] # Assuming 'Music' was the first and dropped
}

st.title('Prediksi Klaster Gen Z Mental Wellness')
st.write('Aplikasi ini memprediksi klaster mental wellness seseorang berdasarkan input yang diberikan.')

st.sidebar.header('Input Data Individu')

# Collect user input for each feature
def user_input_features():
    # Numerical features
    age = st.sidebar.slider('Age', 15, 30, 20)
    daily_social_media_hours = st.sidebar.slider('Daily Social Media Hours', 0.0, 10.0, 3.0)
    screen_time_hours = st.sidebar.slider('Screen Time Hours', 0.0, 15.0, 7.0)
    night_scrolling_frequency = st.sidebar.slider('Night Scrolling Frequency', 0, 7, 3)
    online_gaming_hours = st.sidebar.slider('Online Gaming Hours', 0.0, 10.0, 2.0)
    exercise_frequency_per_week = st.sidebar.slider('Exercise Frequency per Week', 0, 7, 3)
    daily_sleep_hours = st.sidebar.slider('Daily Sleep Hours', 0.0, 12.0, 7.0)
    caffeine_intake_cups = st.sidebar.slider('Caffeine Intake Cups', 0, 5, 1)
    study_work_hours_per_day = st.sidebar.slider('Study/Work Hours per Day', 0.0, 12.0, 6.0)
    overthinking_score = st.sidebar.slider('Overthinking Score (1-10)', 1, 10, 5)
    anxiety_score = st.sidebar.slider('Anxiety Score (1-10)', 1, 10, 5)
    mood_stability_score = st.sidebar.slider('Mood Stability Score (1-10)', 1, 10, 5)
    social_comparison_index = st.sidebar.slider('Social Comparison Index (1-10)', 1, 10, 5)
    sleep_quality_score = st.sidebar.slider('Sleep Quality Score (1-10)', 1, 10, 5)
    motivation_level = st.sidebar.slider('Motivation Level (1-10)', 1, 10, 5)
    emotional_fatigue_score = st.sidebar.slider('Emotional Fatigue Score (1-10)', 1, 10, 5)
    wellbeing_index = st.sidebar.slider('Wellbeing Index (1-10)', 1, 10, 5)

    # Categorical features
    gender = st.sidebar.selectbox('Gender', categorical_cols_info['Gender'] + ['Prefer not to say'])
    country = st.sidebar.selectbox('Country', ['Indonesia'] + categorical_cols_info['Country'])
    student_working_status = st.sidebar.selectbox('Student or Working Status', ['Student'] + categorical_cols_info['Student_Working_Status'])
    content_type_preference = st.sidebar.selectbox('Content Type Preference', ['Music'] + categorical_cols_info['Content_Type_Preference'])

    data = {
        'Age': age,
        'Daily_Social_Media_Hours': daily_social_media_hours,
        'Screen_Time_Hours': screen_time_hours,
        'Night_Scrolling_Frequency': night_scrolling_frequency,
        'Online_Gaming_Hours': online_gaming_hours,
        'Exercise_Frequency_per_Week': exercise_frequency_per_week,
        'Daily_Sleep_Hours': daily_sleep_hours,
        'Caffeine_Intake_Cups': caffeine_intake_cups,
        'Study_Work_Hours_per_Day': study_work_hours_per_day,
        'Overthinking_Score': overthinking_score,
        'Anxiety_Score': anxiety_score,
        'Mood_Stability_Score': mood_stability_score,
        'Social_Comparison_Index': social_comparison_index,
        'Sleep_Quality_Score': sleep_quality_score,
        'Motivation_Level': motivation_level,
        'Emotional_Fatigue_Score': emotional_fatigue_score,
        'Wellbeing_Index': wellbeing_index,
        'Gender': gender,
        'Country': country,
        'Student_Working_Status': student_working_status,
        'Content_Type_Preference': content_type_preference
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('Input Data Anda')
st.write(input_df)

if st.sidebar.button('Prediksi Klaster'):
    # --- Preprocessing the input data ---
    # 1. Apply one-hot encoding for categorical features
    # Create a DataFrame with all possible one-hot encoded columns (including those dropped by drop_first=True)
    # This ensures consistency even if input doesn't have a specific category
    processed_input = input_df.copy()
    
    # Manual one-hot encoding to handle drop_first=True and unseen categories gracefully
    # This assumes the original order of categories in `df` was consistent.
    # For 'Gender', 'Female' is first, so 'Male' will be `Gender_Male`.
    # For 'Country', 'Indonesia' is first, so 'India', 'Other' become `Country_India`, `Country_Other`.
    # For 'Student_Working_Status', 'Student' is first, so 'Working' becomes `Student_Working_Status_Working`.
    # For 'Content_Type_Preference', 'Music' is first, so others become `Content_Type_Preference_Educational`, etc.

    # Gender
    processed_input['Gender_Male'] = (processed_input['Gender'] == 'Male').astype(int)
    processed_input = processed_input.drop(columns=['Gender'])

    # Country
    processed_input['Country_India'] = (processed_input['Country'] == 'India').astype(int)
    processed_input['Country_Other'] = (processed_input['Country'] == 'Other').astype(int)
    processed_input = processed_input.drop(columns=['Country'])

    # Student_Working_Status
    processed_input['Student_Working_Status_Working'] = (processed_input['Student_Working_Status'] == 'Working').astype(int)
    processed_input = processed_input.drop(columns=['Student_Working_Status'])

    # Content_Type_Preference
    for category in ['Educational', 'Entertainment', 'News', 'Social Media', 'Sports']:
        processed_input[f'Content_Type_Preference_{category}'] = (processed_input['Content_Type_Preference'] == category).astype(int)
    processed_input = processed_input.drop(columns=['Content_Type_Preference'])

    # 2. Scale the numerical features
    input_scaled = processed_input.copy()
    input_scaled[numerical_cols_for_app] = scaler.transform(processed_input[numerical_cols_for_app])

    # 3. Apply PCA transformation
    input_pca = pca.transform(input_scaled)

    # 4. Predict the cluster
    predicted_cluster = kmeans_model.predict(input_pca)[0]

    st.subheader('Hasil Prediksi Klaster')
    st.write(f'Individu ini diprediksi berada di Klaster: **{predicted_cluster}**')

    st.info("Catatan: Implementasi one-hot encoding di aplikasi ini menyederhanakan 'drop_first=True' yang digunakan saat pelatihan. Pastikan konsistensi dengan fitur yang digunakan saat pelatihan model.")

