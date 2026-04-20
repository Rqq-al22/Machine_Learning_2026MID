import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist

# --- 1. Definisi Fitur dan Dataframe Awal ---
# Menggunakan df_scaled dan df_selected dari notebook untuk konsistensi
# Dalam aplikasi Streamlit sungguhan, Anda akan menyimpan scaler dan model
# Lalu memuatnya. Untuk demonstrasi, kita akan melatih ulang di sini.

# Memuat ulang data awal untuk mendapatkan fitur yang dipilih (df_selected)
import os
import pandas as pd

base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, 'wellnes1.csv')

df_original = pd.read_csv(file_path, sep=';')
selected_features = ['stress_level_0_10', 'productivity_0_100', 'sleep_quality_1_5', 'screen_time_hours']
all_features_for_comparison = selected_features + ['mental_wellness_index_0_100']

# Menyiapkan Scaler
scaler = StandardScaler()
df_selected_for_scaler = df_original[selected_features]
scaler.fit(df_selected_for_scaler)

# Menyiapkan data yang sudah diskalakan (df_scaled)
df_scaled = pd.DataFrame(scaler.transform(df_selected_for_scaler), columns=selected_features)

# Menjalankan Agglomerative Clustering dengan K=3, linkage='average'
agg_clustering_k3 = AgglomerativeClustering(n_clusters=3, linkage='average')
cluster_labels_for_centroids = agg_clustering_k3.fit_predict(df_scaled)

# Menambahkan label klaster ke df_original untuk inferensi
df_original['cluster_agglomerative_k3'] = cluster_labels_for_centroids

# Menghitung centroid untuk setiap klaster
centroids = []
for i in range(3):
    cluster_points = df_scaled[cluster_labels_for_centroids == i]
    if not cluster_points.empty:
        centroids.append(cluster_points.mean(axis=0))
    else:
        # Handle case where a cluster might be empty (unlikely with this data/K)
        centroids.append(np.zeros(len(selected_features))) # Placeholder
centroids = np.array(centroids)

# Menghitung ringkasan klaster (mean values) dan overall means untuk perbandingan
cluster_agg_k3_summary = df_original.groupby('cluster_agglomerative_k3')[all_features_for_comparison].mean().round(2)
overall_means = df_original[all_features_for_comparison].mean().round(2)

# --- 2. Fungsi Prediksi ---
def predict_agglomerative(data_point_scaled, centroids):
    data_point_scaled = np.array(data_point_scaled).reshape(1, -1)
    distances = cdist(data_point_scaled, centroids)
    cluster = distances.argmin()
    min_distance = distances.min()
    return cluster, min_distance

# --- 3. Aplikasi Streamlit ---
st.set_page_config(layout="wide")
st.title("Aplikasi Prediksi Klaster Kesejahteraan Mental (Agglomerative K=3)")

st.markdown(
    """
    Aplikasi ini menggunakan model **Agglomerative Clustering** dengan **3 klaster** dan metode **'average' linkage** 
    untuk mengkategorikan individu berdasarkan metrik kesejahteraan mental tertentu. 
    
    Silakan masukkan nilai untuk fitur-fitur di bawah ini untuk melihat ke klaster mana data Anda akan diklasifikasikan.
    """
)

st.subheader("Penjelasan Fitur dan Rentang Nilai")
st.markdown(
    """
    Memahami setiap fitur membantu Anda memasukkan data yang akurat. Rentang nilai di bawah didasarkan pada data pelatihan:
    
    *   **Tingkat Stres (0-10):** Tingkat stres yang dilaporkan. 0 = sangat rendah, 10 = sangat tinggi. (Rata-rata data: {overall_means['stress_level_0_10']:.2f})
    *   **Produktivitas (0-100):** Tingkat produktivitas yang dirasakan. 0 = sangat tidak produktif, 100 = sangat produktif. (Rata-rata data: {overall_means['productivity_0_100']:.2f})
    *   **Kualitas Tidur (1-5):** Kualitas tidur. 1 = sangat buruk, 5 = sangat baik. (Rata-rata data: {overall_means['sleep_quality_1_5']:.2f})
    *   **Waktu Layar (jam):** Total waktu yang dihabiskan di depan layar per hari. (Rata-rata data: {overall_means['screen_time_hours']:.2f} jam)
    """
)

st.subheader("Masukkan Data Anda")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        stress_level = st.slider("Tingkat Stres (0-10)", min_value=0.0, max_value=10.0, value=overall_means['stress_level_0_10'], step=0.1)
        productivity = st.slider("Produktivitas (0-100)", min_value=0.0, max_value=100.0, value=overall_means['productivity_0_100'], step=1.0)
    with col2:
        sleep_quality = st.slider("Kualitas Tidur (1-5)", min_value=1.0, max_value=5.0, value=overall_means['sleep_quality_1_5'], step=0.1)
        screen_time = st.slider("Waktu Layar (jam)", min_value=0.0, max_value=24.0, value=overall_means['screen_time_hours'], step=0.1)
    
    submitted = st.form_submit_button("Prediksi Klaster")

    if submitted:
        new_data = np.array([[stress_level, productivity, sleep_quality, screen_time]])
        new_data_scaled = scaler.transform(new_data)
        
        predicted_cluster, dist_to_centroid = predict_agglomerative(new_data_scaled[0], centroids)
        
        st.subheader("\nHasil Prediksi Klaster:")
        st.success(f"Data Anda kemungkinan besar termasuk dalam **Klaster {predicted_cluster}**")
        st.info(f"(Jarak ke centroid klaster: {dist_to_centroid:.2f})")

        st.subheader(f"\nInterpretasi Klaster {predicted_cluster}:")
        cluster_info = cluster_agg_k3_summary.loc[predicted_cluster]
        overall_info = overall_means

        st.markdown(f"**\n--- Klaster {predicted_cluster} ---**")
        st.write(f"*   **Tingkat Stres Rata-rata:** {cluster_info['stress_level_0_10']:.2f}")
        st.write(f"*   **Produktivitas Rata-rata:** {cluster_info['productivity_0_100']:.2f}")
        st.write(f"*   **Kualitas Tidur Rata-rata (1-5):** {cluster_info['sleep_quality_1_5']:.2f}")
        st.write(f"*   **Waktu Layar Rata-rata (jam):** {cluster_info['screen_time_hours']:.2f}")
        st.write(f"*   **Indeks Kesejahteraan Mental Rata-rata (0-100):** {cluster_info['mental_wellness_index_0_100']:.2f}")

        st.markdown("**\nPerbandingan dengan Rata-rata Keseluruhan:**")
        for feature in all_features_for_comparison:
            diff = cluster_info[feature] - overall_info[feature]
            feature_name_display = feature.replace('_', ' ').title().replace('0 10', '0-10').replace('1 5', '1-5').replace('0 100', '0-100')
            if diff > 0:
                st.write(f"-   {feature_name_display}: {cluster_info[feature]:.2f} (Lebih tinggi dari rata-rata keseluruhan sebesar {abs(diff):.2f})")
            elif diff < 0:
                st.write(f"-   {feature_name_display}: {cluster_info[feature]:.2f} (Lebih rendah dari rata-rata keseluruhan sebesar {abs(diff):.2f})")
            else:
                st.write(f"-   {feature_name_display}: {cluster_info[feature]:.2f} (Sama dengan rata-rata keseluruhan)")

        # Detailed interpretation from the notebook
        st.markdown("**\nInterpretasi Mendalam:**")
        if predicted_cluster == 0:
            st.markdown("**Klaster 0: Kelompok Stres Tinggi & Kesejahteraan Mental Rendah**")
            st.write("Klaster ini mengidentifikasi individu yang menghadapi tantangan signifikan dalam kesehatan mental mereka, ditandai dengan stres ekstrem, produktivitas yang terganggu, kualitas tidur yang sangat buruk, dan waktu layar yang tinggi. Kombinasi faktor-faktor ini menciptakan lingkaran setan yang memperparah kondisi mental mereka.")
        elif predicted_cluster == 1:
            st.markdown("**Klaster 1: Kelompok Sehat Mental dan Berprestasi**")
            st.write("Kelompok ini merepresentasikan individu dengan kesehatan mental yang sangat baik, didukung oleh manajemen stres yang efektif, tingkat produktivitas yang luar biasa, kualitas tidur yang optimal, dan waktu layar yang rendah. Mereka adalah contoh bagaimana faktor-faktor ini saling mendukung untuk mencapai kesejahteraan mental yang tinggi.")
        elif predicted_cluster == 2:
            st.markdown("**Klaster 2: Kelompok Produktif Moderat**")
            st.write("Kelaster ini mungkin terdiri dari individu yang produktif namun masih bergumul dengan stres yang berdampak pada kualitas tidur dan secara tidak langsung mempengaruhi kesehatan mental mereka secara keseluruhan.")
