import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')




df5=pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\Skincare Recommendation\final_skincare_dataset.csv')

binary_cols = [
    'Sensitive', 'Combination', 'Oily', 'Dry', 'Normal',
    'Acne-Free', 'Anti-Aging', 'Black-Spot', 'Brightening',
    'Hydrating', 'Moisturizing', 'Oil-Control', 'Pore-Care', 'UV-Protection'
]

df5.columns = df5.columns.str.strip().str.lower().str.replace('-', '_')

binary_cols = [
    'sensitive', 'combination', 'oily', 'dry', 'normal',
    'acne_free', 'anti_aging', 'black_spot', 'brightening',
    'hydrating', 'moisturizing', 'oil_control', 'pore_care', 'uv_protection'
]

#Univariate Analysis
for col in ['product_type'] + binary_cols:
    if col in df5.columns:
        plt.figure(figsize=(6, 4))
        df5[col].value_counts(dropna=False).plot(kind='bar')
        plt.title(f'Distribution of {col}')
        plt.ylabel("Count")
        plt.xlabel(col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print(f"Column not found: {col}")

#  Bivariate Analysis: Effect vs Product Type
for col in binary_cols:
    if col in df5.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df5, x='product_type', hue=col)
        plt.title(f'{col} by Product Type')
        plt.ylabel("Count")
        plt.xlabel("Product Type")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print(f" Column not found: {col}")

correlation_matrix = df5.corr(numeric_only=True)

plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f")
plt.title("Correlation Matrix of All Numeric/Binary Columns")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()