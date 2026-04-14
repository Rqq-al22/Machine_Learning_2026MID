# -*- coding: utf-8 -*-
"""
Streamlit App for Wellness Clustering Analysis
Menggunakan fungsi dari ml_clustering_ipyb.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Import dari file model clustering
try:
    from ml_clustering_ipyb import WellnessClustering
    USING_MODULE = True
except ImportError:
    USING_MODULE = False
    st.warning("Module 'ml_clustering_ipyb.py' not found. Using standalone mode.")

# Set page config
st.set_page_config(
    page_title="Wellness Clustering Analysis",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Wellness Data Clustering Analysis")
st.markdown("Analisis clustering untuk data wellness menggunakan K-Means, Agglomerative, dan DBSCAN")
st.caption(f"Mode: {'Menggunakan ml_clustering_ipyb.py' if USING_MODULE else 'Standalone'}")

# Sidebar
st.sidebar.header("⚙️ Settings")

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file (wellnes1.csv)",
    type=['csv'],
    help="Upload file CSV dengan separator ';'"
)

# Manual file path input
file_path = st.sidebar.text_input(
    "Atau masukkan path file CSV",
    value="wellnes1.csv",
    help="Path ke file CSV"
)

# Initialize clustering object
@st.cache_resource
def init_clustering(file, path):
    try:
        if file is not None:
            df = pd.read_csv(file, sep=';')
        else:
            df = pd.read_csv(path, sep=';')
        
        if USING_MODULE:
            return WellnessClustering(df=df)
        else:
            return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data
clustering_obj = init_clustering(uploaded_file, file_path)

if clustering_obj is not None:
    if USING_MODULE:
        df = clustering_obj.df
    else:
        df = clustering_obj
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Data Overview",
        "🔍 K-Means Clustering",
        "🌳 Agglomerative Clustering",
        "📍 DBSCAN Clustering",
        "📈 Comparison & Insights"
    ])

    # ==================== TAB 1: DATA OVERVIEW ====================
    with tab1:
        st.header("Data Overview")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", df.shape[0])
            st.metric("Columns", df.shape[1])
        with col2:
            st.metric("Missing Values", df.isnull().sum().sum())
            st.metric("Complete Rows", df.dropna().shape[0])
        
        st.subheader("First 5 Rows")
        st.dataframe(df.head())
        
        st.subheader("Dataset Information")
        col_info, col_stats = st.columns(2)
        with col_info:
            st.write("**Data Types:**")
            st.dataframe(df.dtypes.reset_index().rename(
                columns={'index': 'Column', 0: 'Dtype'}
            ))
        with col_stats:
            st.write("**Descriptive Statistics:**")
            st.dataframe(df.describe())
        
        st.subheader("Missing Values")
        missing_df = pd.DataFrame({
            'Missing Count': df.isnull().sum(),
            'Percentage (%)': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(missing_df[missing_df['Missing Count'] > 0])
        
        st.subheader("Box Plots - Feature Distributions")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, col in enumerate(df.columns):
            if i < len(axes):
                sns.boxplot(y=df[col], ax=axes[i], color='skyblue')
                axes[i].set_title(f'Box Plot of {col}')
                axes[i].set_ylabel('')
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, ax=ax)
        ax.set_title('Correlation Heatmap')
        st.pyplot(fig)
        
        st.subheader("Feature Scaling")
        scaler = StandardScaler()
        selected_features = ['stress_level_0_10', 'productivity_0_100', 'sleep_quality_1_5', 'screen_time_hours']
        df_selected = df[selected_features]
        df_scaled = scaler.fit_transform(df_selected)
        df_scaled = pd.DataFrame(df_scaled, columns=selected_features)
        st.write("**Scaled Data (first 5 rows):**")
        st.dataframe(df_scaled.head())
        
        # Store in session state
        st.session_state['df_scaled'] = df_scaled
        st.session_state['df_original'] = df
        st.session_state['scaler'] = scaler
        st.session_state['selected_features'] = selected_features

    # ==================== TAB 2: K-MEANS CLUSTERING ====================
    with tab2:
        st.header("K-Means Clustering")
        
        df_scaled = st.session_state['df_scaled']
        df_original = st.session_state['df_original']
        
        # Elbow Method
        st.subheader("Elbow Method untuk Menentukan Optimal K")
        k_range = range(2, 11)
        wcss = []
        
        with st.spinner("Menghitung WCSS..."):
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(df_scaled)
                wcss.append(kmeans.inertia_)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_range, wcss, marker='o', linestyle='--')
        ax.set_title('Elbow Method for Optimal K')
        ax.set_xlabel('Number of Clusters (K)')
        ax.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
        ax.grid(True)
        st.pyplot(fig)
        
        # Silhouette Scores
        st.subheader("Silhouette Scores")
        silhouette_scores = []
        
        with st.spinner("Menghitung Silhouette Scores..."):
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(df_scaled)
                score = silhouette_score(df_scaled, labels)
                silhouette_scores.append(score)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_range, silhouette_scores, marker='o', linestyle='--', color='green')
        ax.set_title('Silhouette Scores for Optimal K')
        ax.set_xlabel('Number of Clusters (K)')
        ax.set_ylabel('Silhouette Score')
        ax.grid(True)
        st.pyplot(fig)
        
        # Optimal K selection
        st.subheader("Pilih K untuk Clustering")
        optimal_k = st.slider("Number of Clusters (K)", 2, 10, 4)
        
        # Run K-Means
        if st.button("Run K-Means Clustering", type="primary"):
            with st.spinner("Running K-Means..."):
                if USING_MODULE:
                    clustering_obj.scale_features(st.session_state['selected_features'])
                    labels, score = clustering_obj.run_kmeans(n_clusters=optimal_k)
                    df_result = clustering_obj.df.copy()
                else:
                    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(df_scaled)
                    df_result = df_original.copy()
                    df_result['cluster_kmeans'] = labels
                    score = silhouette_score(df_scaled, labels)
                
                st.session_state['kmeans_result'] = df_result
                st.session_state['kmeans_labels'] = labels
                st.metric("Final Silhouette Score", f"{score:.4f}")
                
                # PCA Visualization
                st.subheader("Visualization with PCA")
                pca = PCA(n_components=2)
                df_pca = pca.fit_transform(df_scaled)
                df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])
                df_pca['cluster'] = labels
                
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(df_pca['PC1'], df_pca['PC2'], 
                                    c=df_pca['cluster'], cmap='viridis', 
                                    s=100, alpha=0.7, edgecolor='w')
                ax.set_title(f'K-Means Clustering (K={optimal_k}) Visualization with PCA')
                ax.set_xlabel('Principal Component 1')
                ax.set_ylabel('Principal Component 2')
                plt.colorbar(scatter, label='Cluster')
                ax.grid(True)
                st.pyplot(fig)
                
                # Cluster Analysis
                st.subheader("Cluster Analysis")
                features = st.session_state['selected_features'] + ['mental_wellness_index_0_100']
                cluster_summary = df_result.groupby('cluster_kmeans')[features].mean().round(2)
                st.dataframe(cluster_summary)

    # ==================== TAB 3: AGGLOMERATIVE CLUSTERING ====================
    with tab3:
        st.header("Agglomerative Clustering")
        
        df_scaled = st.session_state['df_scaled']
        df_original = st.session_state['df_original']
        
        # Parameter selection
        st.subheader("Parameter Selection")
        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.slider("Number of Clusters", 2, 10, 2, key="agg_k")
        with col2:
            linkage = st.selectbox("Linkage Method", ['ward', 'complete', 'average'], key="agg_linkage")
        
        # Find best parameters
        if st.button("Find Best Parameters", key="agg_find"):
            st.subheader("Parameter Search Results")
            k_range = range(2, 11)
            linkage_methods = ['ward', 'complete', 'average']
            results = []
            
            with st.spinner("Searching for best parameters..."):
                for link in linkage_methods:
                    for k in k_range:
                        if link == 'ward' and k == 1:
                            continue
                        agg = AgglomerativeClustering(n_clusters=k, linkage=link)
                        labels = agg.fit_predict(df_scaled)
                        if len(set(labels)) > 1:
                            score = silhouette_score(df_scaled, labels)
                            results.append({
                                'Linkage': link,
                                'K': k,
                                'Silhouette Score': score
                            })
            
            results_df = pd.DataFrame(results)
            if not results_df.empty:
                st.dataframe(results_df.sort_values('Silhouette Score', ascending=False))
                best = results_df.loc[results_df['Silhouette Score'].idxmax()]
                st.success(f"Best: K={int(best['K'])}, Linkage={best['Linkage']}, Score={best['Silhouette Score']:.4f}")
        
        # Run Agglomerative
        if st.button("Run Agglomerative Clustering", type="primary", key="agg_run"):
            with st.spinner("Running Agglomerative Clustering..."):
                if USING_MODULE:
                    clustering_obj.scale_features(st.session_state['selected_features'])
                    labels, score = clustering_obj.run_agglomerative(n_clusters=n_clusters, linkage=linkage)
                    df_result = clustering_obj.df.copy()
                else:
                    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
                    labels = agg.fit_predict(df_scaled)
                    df_result = df_original.copy()
                    df_result['cluster_agglomerative'] = labels
                    score = silhouette_score(df_scaled, labels)
                
                st.session_state['agg_result'] = df_result
                st.metric("Silhouette Score", f"{score:.4f}")
                
                # PCA Visualization
                pca = PCA(n_components=2)
                df_pca = pca.fit_transform(df_scaled)
                df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])
                df_pca['cluster'] = labels
                
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(df_pca['PC1'], df_pca['PC2'], 
                                    c=df_pca['cluster'], cmap='viridis', 
                                    s=100, alpha=0.7, edgecolor='w')
                ax.set_title(f'Agglomerative Clustering (K={n_clusters}, Linkage={linkage})')
                ax.set_xlabel('Principal Component 1')
                ax.set_ylabel('Principal Component 2')
                plt.colorbar(scatter, label='Cluster')
                ax.grid(True)
                st.pyplot(fig)
                
                # Cluster Analysis
                st.subheader("Cluster Analysis")
                features = st.session_state['selected_features'] + ['mental_wellness_index_0_100']
                cluster_summary = df_result.groupby('cluster_agglomerative')[features].mean().round(2)
                st.dataframe(cluster_summary)

    # ==================== TAB 4: DBSCAN CLUSTERING ====================
    with tab4:
        st.header("DBSCAN Clustering")
        
        df_scaled = st.session_state['df_scaled']
        df_original = st.session_state['df_original']
        
        # Parameter selection
        st.subheader("Parameter Selection")
        col1, col2 = st.columns(2)
        with col1:
            eps = st.slider("Epsilon (eps)", 0.1, 2.0, 0.4, 0.05, key="dbscan_eps")
        with col2:
            min_samples = st.slider("Min Samples", 2, 20, 10, key="dbscan_min")
        
        # Find best parameters
        if st.button("Find Best Parameters", key="dbscan_find"):
            st.subheader("Parameter Search Results")
            eps_range = np.arange(0.3, 1.6, 0.1)
            min_samples_range = range(3, 11)
            results = []
            
            with st.spinner("Searching for best parameters..."):
                for e in eps_range:
                    for m in min_samples_range:
                        dbscan = DBSCAN(eps=e, min_samples=m)
                        labels = dbscan.fit_predict(df_scaled)
                        unique_labels = set(labels)
                        if -1 in unique_labels:
                            unique_labels.remove(-1)
                        if len(unique_labels) > 1:
                            non_noise = labels != -1
                            if sum(non_noise) > 1:
                                score = silhouette_score(df_scaled[non_noise], labels[non_noise])
                                results.append({
                                    'Eps': round(e, 1),
                                    'Min Samples': m,
                                    'Silhouette Score': score,
                                    'Clusters': len(unique_labels),
                                    'Noise Points': sum(labels == -1)
                                })
            
            results_df = pd.DataFrame(results)
            if not results_df.empty:
                st.dataframe(results_df.sort_values('Silhouette Score', ascending=False))
                best = results_df.loc[results_df['Silhouette Score'].idxmax()]
                st.success(f"Best: eps={best['Eps']}, min_samples={int(best['Min Samples'])}, Score={best['Silhouette Score']:.4f}")
        
        # Run DBSCAN
        if st.button("Run DBSCAN Clustering", type="primary", key="dbscan_run"):
            with st.spinner("Running DBSCAN..."):
                if USING_MODULE:
                    clustering_obj.scale_features(st.session_state['selected_features'])
                    labels, score, n_clusters, n_noise = clustering_obj.run_dbscan(eps=eps, min_samples=min_samples)
                    df_result = clustering_obj.df.copy()
                else:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(df_scaled)
                    df_result = df_original.copy()
                    df_result['cluster_dbscan'] = labels
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = sum(labels == -1)
                    if n_clusters > 1:
                        non_noise = labels != -1
                        if sum(non_noise) > 1:
                            score = silhouette_score(df_scaled[non_noise], labels[non_noise])
                        else:
                            score = None
                    else:
                        score = None
                
                st.session_state['dbscan_result'] = df_result
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Number of Clusters", n_clusters)
                col2.metric("Noise Points", n_noise)
                col3.metric("Noise Percentage", f"{(n_noise/len(labels)*100):.1f}%")
                
                if score:
                    st.metric("Silhouette Score (excluding noise)", f"{score:.4f}")
                
                # PCA Visualization
                pca = PCA(n_components=2)
                df_pca = pca.fit_transform(df_scaled)
                df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])
                df_pca['cluster'] = labels
                
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(df_pca['PC1'], df_pca['PC2'], 
                                    c=df_pca['cluster'], cmap='tab10', 
                                    s=100, alpha=0.7, edgecolor='w')
                ax.set_title(f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})')
                ax.set_xlabel('Principal Component 1')
                ax.set_ylabel('Principal Component 2')
                plt.colorbar(scatter, label='Cluster (-1 = Noise)')
                ax.grid(True)
                st.pyplot(fig)

    # ==================== TAB 5: COMPARISON & INSIGHTS ====================
    with tab5:
        st.header("Model Comparison & Insights")
        
        has_kmeans = 'kmeans_result' in st.session_state
        has_agg = 'agg_result' in st.session_state
        has_dbscan = 'dbscan_result' in st.session_state
        
        if has_kmeans or has_agg or has_dbscan:
            st.subheader("Best Model Comparison")
            
            comparison_data = []
            df_scaled = st.session_state['df_scaled']
            
            if has_kmeans:
                score = silhouette_score(df_scaled, st.session_state['kmeans_labels'])
                comparison_data.append({
                    'Model': 'K-Means',
                    'Silhouette Score': f"{score:.4f}",
                    'Clusters': len(set(st.session_state['kmeans_labels']))
                })
            
            if has_agg:
                agg_labels = st.session_state['agg_result']['cluster_agglomerative']
                if len(set(agg_labels)) > 1:
                    score = silhouette_score(df_scaled, agg_labels)
                    comparison_data.append({
                        'Model': 'Agglomerative',
                        'Silhouette Score': f"{score:.4f}",
                        'Clusters': len(set(agg_labels))
                    })
            
            if has_dbscan:
                dbscan_labels = st.session_state['dbscan_result']['cluster_dbscan']
                if len(set(dbscan_labels)) > 2:
                    non_noise = dbscan_labels != -1
                    if sum(non_noise) > 1:
                        score = silhouette_score(df_scaled[non_noise], dbscan_labels[non_noise])
                        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
                        comparison_data.append({
                            'Model': 'DBSCAN',
                            'Silhouette Score': f"{score:.4f}",
                            'Clusters': n_clusters
                        })
            
            if comparison_data:
                st.table(pd.DataFrame(comparison_data))
            
            # Final Insights
            st.subheader("Key Insights")
            st.markdown("""
            ### Hubungan Antar Faktor dengan Kesehatan Mental
            
            1. **Stress Level**:
               - Stress yang tinggi berkorelasi kuat dengan mental wellness yang rendah
               - Kelompok dengan stress > 7 umumnya memiliki mental wellness index < 30
            
            2. **Productivity**:
               - Produktivitas tinggi (>70) biasanya diikuti mental wellness yang baik
               - Produktivitas rendah (<40) menjadi indikator potensi masalah mental
            
            3. **Sleep Quality**:
               - Kualitas tidur yang buruk (skor 1-2) sangat mempengaruhi kesehatan mental
               - Tidur yang cukup dan berkualitas adalah prediktor kuat wellness
            
            4. **Screen Time**:
               - Screen time berlebihan (>8 jam) berkorelasi dengan stress yang lebih tinggi
            
            ### Rekomendasi Intervensi
            
            - **High Risk Group**: Fokus pada manajemen stress dan perbaikan kualitas tidur
            - **Moderate Group**: Tingkatkan produktivitas dan jaga keseimbangan screen time
            - **Wellness Group**: Pertahankan kebiasaan baik dan jadikan role model
            """)
        else:
            st.info("Jalankan salah satu model clustering terlebih dahulu untuk melihat perbandingan")

else:
    st.error("Tidak dapat memuat data. Silakan periksa path file atau upload file CSV.")
