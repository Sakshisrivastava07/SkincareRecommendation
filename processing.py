import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('C:/Users/HP/Downloads/export_skincare.csv')
#print(df.head())
df1=df.copy()
df2 = df1.copy()
df2['notable_effects'] = df2['notable_effects'].fillna('')
df2['notable_effects_list'] = df2['notable_effects'].apply(lambda x: [i.strip() for i in x.split(',')])
from itertools import chain
all_effects = sorted(set(chain.from_iterable(df2['notable_effects_list'])))

for effect in all_effects:
    df2[effect] = df2['notable_effects_list'].apply(lambda x: 1 if effect in x else 0)


df2.drop(columns=['notable_effects_list'], inplace=True)


df2['skintype'] = df2['skintype'].fillna('')
skin_types = ['Sensitive', 'Combination', 'Oily', 'Dry', 'Normal']

for skin in skin_types:
    df2[skin] = df2['skintype'].apply(lambda x: 1 if skin in x else 0)

df2['notable_effects'] = df2['notable_effects'].fillna('')
df2['notable_effects_list'] = df2['notable_effects'].apply(lambda x: [i.strip() for i in x.split(',')])

df2['skintype'] = df2['skintype'].fillna('')
df2['skintype_list'] = df2['skintype'].apply(lambda x: [i.strip() for i in x.split(',')])

from itertools import chain
unique_effects = sorted(set(chain.from_iterable(df2['notable_effects_list'])))
unique_skin_types = sorted(set(chain.from_iterable(df2['skintype_list'])))

for effect in unique_effects:
    df2[effect] = df2['notable_effects_list'].apply(lambda x: 1 if effect in x else 0)

for skin in unique_skin_types:
    df2[skin] = df2['skintype_list'].apply(lambda x: 1 if skin in x else 0)

column_names_lower = df2.columns.str.lower()
duplicates = df2.columns[column_names_lower.duplicated(keep=False)]
actual_duplicates = df2.loc[:, duplicates]
if "Oil-Control" in df2.columns and "Oil-control" in df2.columns:
    df2["Oil-Control"] = df2[["Oil-Control", "Oil-control"]].max(axis=1)
    df2.drop(columns=["Oil-control"], inplace=True)

# print(df2.head())

df3=df2.copy()
#print(df2['product_type'].value_counts())

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df2['product_type_encoded'] = label_encoder.fit_transform(df2['product_type'])
product_type_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# print(df2.head())

df2.to_csv('df2_final.csv', index=False)
from IPython.display import FileLink
FileLink('df2_final.csv')
df2 = pd.read_csv("df2_final.csv")
np.random.seed(42)
columns = df2.columns
num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df2.select_dtypes(exclude=[np.number]).columns.tolist()

dummy_data = {}
for col in columns:
    if col in num_cols:
        dummy_data[col] = np.random.choice([0, 1, np.nan], size=3000, p=[0.45, 0.45, 0.10])
    else:
        existing_values = df2[col].dropna().unique()
        dummy_data[col] = np.random.choice(np.append(existing_values, [np.nan]), size=3000)

df_dummy = pd.DataFrame(dummy_data)

df2_updated = pd.concat([df2, df_dummy], ignore_index=True)
df4=df2_updated.copy()
#print(df4.tail())

if 'Unnamed: 0' in df4.columns:
    df4.drop(columns=['Unnamed: 0'], inplace=True)

if 'No-Whitecast' in df4.columns:
    df4.drop(columns=['No-Whitecast'], inplace=True)

if 'Refreshing' in df4.columns:
    df4.drop(columns=['Refreshing'], inplace=True)

if 'Skin-Barrier' in df4.columns:
    df4.drop(columns=['Skin-Barrier'], inplace=True)

if 'Soothing' in df4.columns:
    df4.drop(columns=['Soothing'], inplace=True)

df5 = df4.dropna(subset=['product_type'])

if 'price' in df5.columns:
    df5.drop(columns=['price'], inplace=True)

if 'description' in df5.columns:
    df5.drop(columns=['description'], inplace=True)

if 'notable_effects' in df5.columns:
    df5.drop(columns=['notable_effects'], inplace=True)

if 'skintype' in df5.columns:
    df5.drop(columns=['skintype'], inplace=True)

if 'labels' in df5.columns:
    df5.drop(columns=['labels'], inplace=True)

if 'Balancing' in df5.columns:
    df5.drop(columns=['Balancing'], inplace=True)


if 'Acne-Spot' in df5.columns:
    df5.drop(columns=['Acne-Spot'], inplace=True)

if 'product_href' in df5.columns:
    df5.drop(columns=['product_href'], inplace=True)

if 'product_name' in df5.columns:
    df5.drop(columns=['product_name'], inplace=True)

if 'brand' in df5.columns:
    df5.drop(columns=['brand'], inplace=True)

if 'picture_src' in df5.columns:
    df5.drop(columns=['picture_src'], inplace=True)

if 'notable_effects_list' in df5.columns:
    df5.drop(columns=['notable_effects_list'], inplace=True)

if 'skintype_list' in df5.columns:
    df5.drop(columns=['skintype_list'], inplace=True)



df5 = df5.dropna(subset=['product_type'])
binary_cols = [
    'Sensitive', 'Combination', 'Oily', 'Dry', 'Normal',
    'Acne-Free', 'Anti-Aging', 'Black-Spot',
    'Brightening', 'Hydrating', 'Moisturizing', 'Oil-Control',
    'Pore-Care', 'UV-Protection'
]

df5[binary_cols] = df5[binary_cols].fillna(0)

#print(df5.isnull().sum())

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df5['product_type_encoded'] = label_encoder.fit_transform(df5['product_type'])
product_type_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

#print(df5.tail())
#print(df5.isnull().sum())

df5.to_csv("final_skincare_dataset.csv", index=False)


# Set random seed for reproducibility
np.random.seed(42)

# Number of synthetic samples to generate
n_samples = 3000

# Unique product types and their encoded values
product_types = df5['product_type'].unique()
product_type_map = dict(zip(product_types, df5['product_type_encoded'].unique()))

# Function to generate synthetic row based on observed probabilities
def generate_row():
    row = {}
    # Randomly choose a product type based on original distribution
    product = np.random.choice(product_types, p=df5['product_type'].value_counts(normalize=True).loc[product_types].values)
    row['product_type'] = product
    row['product_type_encoded'] = product_type_map[product]
    
    # For each binary feature, sample 0 or 1 based on the original column's probability of being 1
    for col in df5.columns:
        if col not in ['product_type', 'product_type_encoded']:
            prob = df5[col].mean()
            row[col] = np.random.choice([0.0, 1.0], p=[1 - prob, prob])
    
    return row

# Generate all rows
synthetic_data = pd.DataFrame([generate_row() for _ in range(n_samples)])

# Reorder columns to match original
synthetic_data = synthetic_data[df5.columns]

# Append new rows to df5
df5 = pd.concat([df5, synthetic_data], ignore_index=True)

# Check shape to confirm rows added
print("Updated df5 shape:", df5.shape)


print(df5.tail())
df5.to_csv("final_skincare_dataset.csv", index=False)
