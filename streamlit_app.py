# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Set page config - HARUS di baris pertama setelah streamlit import
st.set_page_config(
    page_title="Wellness Clustering Analysis",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Wellness Data Clustering Analysis")
st.markdown("Analisis clustering untuk data wellness menggunakan K-Means, Agglomerative, dan DBSCAN")

# Sidebar
st.sidebar.header("⚙️ Settings")

# Cari file CSV di beberapa lokasi
@st.cache_data
def find_and_load_data():
    """Cari file CSV di berbagai lokasi"""
    possible_paths = [
        'data/wellnes1.csv',           # Path yang benar
        'wellnes1.csv',                 # Root folder
        '../data/wellnes1.csv',         # Relative path
        '/content/drive/MyDrive/ML_MID/wellnes1.csv',  # Google Colab path
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, sep=';')
                return df, path
            except Exception as e:
                continue
    
    return None, None

# Upload file option
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file (wellnes1.csv)",
    type=['csv'],
    help="Upload file CSV dengan separator ';'"
)

# Load data
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=';')
        st.sidebar.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        st.stop()
else:
    # Try to find local file
    df, file_path = find_and_load_data()
    if df is not None:
        st.sidebar.success(f"✅ Data loaded from: {file_path} ({df.shape[0]} rows, {df.shape[1]} columns)")
    else:
        st.error("""
        ❌ **Tidak dapat menemukan file data!**
        
        Pastikan file `wellnes1.csv` ada di:
        - Folder `data/wellnes1.csv`
        - Atau upload file melalui tombol di sidebar
        
        **Struktur folder yang benar:**
