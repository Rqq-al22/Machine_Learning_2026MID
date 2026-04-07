import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ======================
# LOAD DATA
# ======================
df = pd.read_csv('genz_dataset.csv')  # pastikan file ada di folder

# Drop kolom tidak perlu
if 'Kaggle' in df.columns:
    df = df.drop(columns=['Kaggle'])

# ======================
# ENCODING
# ======================
categorical_cols = df.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# ======================
# HANDLE OUTLIER (OPSIONAL)
# ======================
for col in df_encoded.columns:
    Q1 = df_encoded[col].quantile(0.25)
    Q3 = df_encoded[col].quantile(0.75)
    IQR = Q3 - Q1
    df_encoded[col] = df_encoded[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

# ======================
# SCALING
# ======================
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded)

# ======================
# PCA
# ======================
pca = PCA(n_components=0.95)
df_pca = pca.fit_transform(df_scaled)

# ======================
# KMEANS
# ======================
kmeans_model = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_model.fit(df_pca)

# ======================
# SAVE SEMUA (PENTING)
# ======================
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca_model.pkl')
joblib.dump(kmeans_model, 'kmeans_model.pkl')

# 🔥 INI PALING PENTING
joblib.dump(df_encoded.columns.tolist(), 'feature_columns.pkl')

print("✅ Semua model berhasil disimpan!")
