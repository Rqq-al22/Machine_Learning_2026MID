# -*- coding: utf-8 -*-
"""ML_CLUSTERING.ipyb - Model Clustering Functions"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

class WellnessClustering:
    """Class untuk menangani semua operasi clustering"""
    
    def __init__(self, file_path=None, df=None, sep=';'):
        if df is not None:
            self.df = df
        elif file_path is not None:
            self.df = pd.read_csv(file_path, sep=sep)
        else:
            raise ValueError("Either file_path or df must be provided")
        
        self.df_scaled = None
        self.scaler = StandardScaler()
        self.selected_features = ['stress_level_0_10', 'productivity_0_100', 
                                   'sleep_quality_1_5', 'screen_time_hours']
    
    def explore_data(self):
        """Return data exploration results"""
        return {
            'head': self.df.head(),
            'info': self.df.dtypes,
            'describe': self.df.describe(),
            'missing': self.df.isnull().sum(),
            'missing_pct': self.df.isnull().sum() / len(self.df) * 100
        }
    
    def scale_features(self, features=None):
        """Scale the selected features"""
        if features:
            self.selected_features = features
        df_selected = self.df[self.selected_features]
        self.df_scaled = self.scaler.fit_transform(df_selected)
        return pd.DataFrame(self.df_scaled, columns=self.selected_features)
    
    def plot_boxplots(self):
        """Generate box plots for all numerical features"""
        numerical_cols = self.df.columns
        num_features = len(numerical_cols)
        num_rows = (num_features + 2) // 3
        fig, axes = plt.subplots(num_rows, 3, figsize=(18, num_rows * 5))
        axes = axes.flatten()
        
        for i, col in enumerate(numerical_cols):
            sns.boxplot(y=self.df[col], ax=axes[i], color='skyblue')
            axes[i].set_title(f'Box Plot of {col}')
            axes[i].set_ylabel('')
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self):
        """Generate correlation heatmap"""
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_matrix = self.df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
                   fmt='.2f', linewidths=.5, ax=ax)
        ax.set_title('Correlation Heatmap of Features')
        return fig
    
    def kmeans_elbow_method(self, k_range=range(2, 11)):
        """Calculate WCSS for elbow method"""
        wcss = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.df_scaled)
            wcss.append(kmeans.inertia_)
        return list(k_range), wcss
    
    def kmeans_silhouette_scores(self, k_range=range(2, 11)):
        """Calculate silhouette scores for different k values"""
        scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.df_scaled)
            score = silhouette_score(self.df_scaled, labels)
            scores.append(score)
        return list(k_range), scores
    
    def run_kmeans(self, n_clusters=4):
        """Run K-Means clustering"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.df_scaled)
        self.df['cluster_kmeans'] = labels
        score = silhouette_score(self.df_scaled, labels)
        return labels, score
    
    def run_agglomerative(self, n_clusters=2, linkage='average'):
        """Run Agglomerative clustering"""
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = agg.fit_predict(self.df_scaled)
        self.df['cluster_agglomerative'] = labels
        score = silhouette_score(self.df_scaled, labels)
        return labels, score
    
    def find_best_agglomerative(self):
        """Find best parameters for Agglomerative clustering"""
        k_range = range(2, 11)
        linkage_methods = ['ward', 'complete', 'average']
        best_score = -1
        best_k = None
        best_linkage = None
        
        for linkage in linkage_methods:
            for k in k_range:
                if linkage == 'ward' and k == 1:
                    continue
                agg = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                labels = agg.fit_predict(self.df_scaled)
                if len(set(labels)) > 1:
                    score = silhouette_score(self.df_scaled, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
                        best_linkage = linkage
        
        return best_k, best_linkage, best_score
    
    def run_dbscan(self, eps=0.4, min_samples=10):
        """Run DBSCAN clustering"""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(self.df_scaled)
        self.df['cluster_dbscan'] = labels
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = sum(labels == -1)
        
        score = None
        if n_clusters > 1:
            non_noise = labels != -1
            if sum(non_noise) > 1:
                score = silhouette_score(self.df_scaled[non_noise], labels[non_noise])
        
        return labels, score, n_clusters, n_noise
    
    def find_best_dbscan(self):
        """Find best parameters for DBSCAN"""
        eps_range = np.arange(0.3, 1.6, 0.1)
        min_samples_range = range(3, 11)
        best_score = -1
        best_eps = None
        best_min_samples = None
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(self.df_scaled)
                unique_labels = set(labels)
                if -1 in unique_labels:
                    unique_labels.remove(-1)
                if len(unique_labels) > 1:
                    non_noise = labels != -1
                    if sum(non_noise) > 1:
                        score = silhouette_score(self.df_scaled[non_noise], 
                                                labels[non_noise])
                        if score > best_score:
                            best_score = score
                            best_eps = eps
                            best_min_samples = min_samples
        
        return best_eps, best_min_samples, best_score
    
    def plot_pca_clusters(self, cluster_labels, title="Clustering Results"):
        """Plot PCA visualization of clusters"""
        pca = PCA(n_components=2)
        df_pca = pca.fit_transform(self.df_scaled)
        df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])
        df_pca['cluster'] = cluster_labels
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(df_pca['PC1'], df_pca['PC2'], 
                           c=df_pca['cluster'], cmap='viridis', 
                           s=100, alpha=0.7, edgecolor='w')
        ax.set_title(title)
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        plt.colorbar(scatter, label='Cluster')
        ax.grid(True)
        return fig
    
    def get_cluster_summary(self, cluster_col):
        """Get mean values per cluster"""
        features = self.selected_features + ['mental_wellness_index_0_100']
        return self.df.groupby(cluster_col)[features].mean().round(2)

# Jika dijalankan sebagai script utama
if __name__ == "__main__":
    # Contoh penggunaan
    wc = WellnessClustering(file_path='/content/drive/MyDrive/ML_MID/wellnes1.csv')
    print("Data loaded successfully!")
    print(f"Shape: {wc.df.shape}")
