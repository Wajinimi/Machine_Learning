import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

load_file = "C:\\Users\\WAJI\\OneDrive\\projects\\PYTHON\\house_price_dataset1.csv"
open_file = pd.read_csv(load_file)
cleaned_data = open_file.dropna(subset=["House_Price"])

pca = PCA(n_components = 2)
principal_components = pca.fit_transform(cleaned_data)
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))
pc_df = pd.DataFrame(data= principal_components, columns =["PC1", "PC2"])
print(pc_df.head())
print(pca.components_)
feature_contributions = pd.DataFrame(
    pca.components_,
    columns=cleaned_data.columns,  # Original feature names
    index=["PC1", "PC2"]       # Name the rows as PC1, PC2, etc.
)
print(feature_contributions)



# Assume your transformed data is in `principal_df`
plt.figure(figsize=(8, 6))
plt.scatter(pc_df['PC1'], pc_df['PC2'], alpha=0.7)
plt.title('2D Projection of Data (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()
