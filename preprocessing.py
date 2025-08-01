import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# %%
df = pd.read_csv('export_skincare.csv') 

# %%
df

# %%
df1=df.copy()

# %%
df1

# %%
# Create a new dataset df2 by copying df22 to preserve original data
df2 = df1.copy()

# Step 1: Ensure 'notable_effects' exists and fill missing values
df2['notable_effects'] = df2['notable_effects'].fillna('')

# Step 2: Split 'notable_effects' into lists
df2['notable_effects_list'] = df2['notable_effects'].apply(lambda x: [i.strip() for i in x.split(',')])

# Step 3: Get unique effect terms
from itertools import chain
all_effects = sorted(set(chain.from_iterable(df2['notable_effects_list'])))

# Step 4: One-hot encode each effect term as a new column (1 if present, 0 otherwise)
for effect in all_effects:
    df2[effect] = df2['notable_effects_list'].apply(lambda x: 1 if effect in x else 0)

# Step 5: Drop only the helper list column (not the original 'notable_effects')
df2.drop(columns=['notable_effects_list'], inplace=True)

# Display the result
#import caas_jupyter_tools as tools;
#tools.display_dataframe_to_user(name="df2 with One-Hot Encoded Notable Effects", dataframe=df2)
df2.head()         # View top 5 rows
  



# %%

df2['skintype'] = df2['skintype'].fillna('')

# Define the expected skin type categories
skin_types = ['Sensitive', 'Combination', 'Oily', 'Dry', 'Normal']

# Create one-hot columns based on whether each skin type is mentioned in the string
for skin in skin_types:
    df2[skin] = df2['skintype'].apply(lambda x: 1 if skin in x else 0)

# Show updated dataframe
df2.head()


# %%
print(df1['skintype'].value_counts())

# %%
df2 = df1.copy()

# Step 2: Clean and split 'notable_effects' into list format
df2['notable_effects'] = df2['notable_effects'].fillna('')
df2['notable_effects_list'] = df2['notable_effects'].apply(lambda x: [i.strip() for i in x.split(',')])

# Step 3: Clean and split 'skintype' into list format
df2['skintype'] = df2['skintype'].fillna('')
df2['skintype_list'] = df2['skintype'].apply(lambda x: [i.strip() for i in x.split(',')])

# Step 4: Get unique tags from both columns
from itertools import chain
unique_effects = sorted(set(chain.from_iterable(df2['notable_effects_list'])))
unique_skin_types = sorted(set(chain.from_iterable(df2['skintype_list'])))

# Step 5: One-hot encode all unique effects
for effect in unique_effects:
    df2[effect] = df2['notable_effects_list'].apply(lambda x: 1 if effect in x else 0)

# Step 6: One-hot encode all unique skin types
for skin in unique_skin_types:
    df2[skin] = df2['skintype_list'].apply(lambda x: 1 if skin in x else 0)

# Step 7: Drop helper list columns only
#df2.drop(columns=['notable_effects_list', 'skintype_list'], inplace=True)

# Show updated df2
#import caas_jupyter_tools as tools; tools.display_dataframe_to_user(name="df2 with One-Hot Encoded Effects and Skin Types", dataframe=df2)
df2


# %%
df2.columns

# %%
column_names_lower = df2.columns.str.lower()
duplicates = df2.columns[column_names_lower.duplicated(keep=False)]
actual_duplicates = df2.loc[:, duplicates]
if "Oil-Control" in df2.columns and "Oil-control" in df2.columns:
    df2["Oil-Control"] = df2[["Oil-Control", "Oil-control"]].max(axis=1)
    df2.drop(columns=["Oil-control"], inplace=True)



# %%
df2.head()

# %% [markdown]
# df2.columns
# 

# %%
df3=df2.copy()

# %%
print(df2['product_type'].value_counts())

# %%
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df2['product_type_encoded'] = label_encoder.fit_transform(df2['product_type'])
product_type_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))


# %%
df2

# %%
df2.head()

# %%
df2.columns

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
effect_cols = [
    'Acne-Free', 'Acne-Spot', 'Anti-Aging', 'Balancing', 'Black-Spot',
    'Brightening', 'Hydrating', 'Moisturizing', 'No-Whitecast',
    'Oil-Control', 'Pore-Care', 'Refreshing', 'Skin-Barrier',
    'Soothing', 'UV-Protection']
grouped = df2.groupby('product_type')[effect_cols].sum()
plt.figure(figsize=(12, 6))
sns.heatmap(grouped, annot=True, cmap="YlGnBu", fmt="d")
plt.title("Notable Effects by Product Type")
plt.xlabel("Notable Effects")
plt.ylabel("Product Type")
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Sum notable effects per product type
effect_cols = [
    'Acne-Free', 'Acne-Spot', 'Anti-Aging', 'Balancing', 'Black-Spot',
    'Brightening', 'Hydrating', 'Moisturizing', 'No-Whitecast',
    'Oil-Control', 'Pore-Care', 'Refreshing', 'Skin-Barrier',
    'Soothing', 'UV-Protection'
]
grouped = df2.groupby('product_type')[effect_cols].sum()

# Plot stacked bar chart
grouped.T.plot(kind='bar', stacked=True, figsize=(14, 6))
plt.title("Notable Effects by Product Type")
plt.ylabel("Number of Products")
plt.xlabel("Notable Effect")
plt.legend(title="Product Type")
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Define columns for notable effects and skintypes
effect_cols = [
    'Acne-Free', 'Acne-Spot', 'Anti-Aging', 'Balancing', 'Black-Spot',
    'Brightening', 'Hydrating', 'Moisturizing', 'No-Whitecast',
    'Oil-Control', 'Pore-Care', 'Refreshing', 'Skin-Barrier',
    'Soothing', 'UV-Protection'
]

skintype_cols = ['Sensitive', 'Combination', 'Oily', 'Dry', 'Normal']

# Sum notable effects for each skin type
grouped_skin = df2[effect_cols + skintype_cols].copy()

# Melt data to long format for better plotting
melted = grouped_skin.melt(id_vars=skintype_cols, value_vars=effect_cols,
                           var_name='Effect', value_name='Present')

# Filter to only rows where the effect is present (value = 1)
melted = melted[melted['Present'] == 1]

# Create long format of skintype and count co-occurrences
melted_skin = melted.melt(id_vars=['Effect'], value_vars=skintype_cols,
                          var_name='SkinType', value_name='HasType')
melted_skin = melted_skin[melted_skin['HasType'] == 1]

# Count number of products with each Effect–SkinType pair
effect_skin_counts = melted_skin.groupby(['Effect', 'SkinType']).size().unstack().fillna(0)

# Plot stacked bar chart
effect_skin_counts.plot(kind='bar', stacked=True, figsize=(14, 6))
plt.title("Notable Effects by Skin Type")
plt.ylabel("Number of Products")
plt.xlabel("Notable Effect")
plt.legend(title="Skin Type")
plt.tight_layout()
plt.show()


# %%
df2.head()

# %%
df2.to_csv('df2_final.csv', index=False)


# %%
from IPython.display import FileLink
FileLink('df2_final.csv')


# %%
df3=df2.copy()

# %%
import pandas as pd
import numpy as np
import random

# Load the original df2 dataset from uploaded file
df2 = pd.read_csv("df2_final.csv")

# Set random seed
np.random.seed(42)

# Get column names
columns = df2.columns
num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df2.select_dtypes(exclude=[np.number]).columns.tolist()

# Generate 3000 dummy rows with unclean data
dummy_data = {}
for col in columns:
    if col in num_cols:
        dummy_data[col] = np.random.choice([0, 1, np.nan], size=3000, p=[0.45, 0.45, 0.10])
    else:
        existing_values = df2[col].dropna().unique()
        dummy_data[col] = np.random.choice(np.append(existing_values, [np.nan]), size=3000)

df_dummy = pd.DataFrame(dummy_data)

# Combine original and dummy datasets
df2_updated = pd.concat([df2, df_dummy], ignore_index=True)

#import caas_jupyter_tools as tools; tools.display_dataframe_to_user(name="df2 with Dummy Rows Added", dataframe=df2_updated)


# %%
df2_updated.tail()

# %%
df4=df2_updated.copy()

# %%
df4.isnull().sum()

# %%
if 'Unnamed: 0' in df4.columns:
    df4.drop(columns=['Unnamed: 0'], inplace=True)
    

# %%
df4

# %%
df4.columns

# %%
if 'No-Whitecast' in df4.columns:
    df4.drop(columns=['No-Whitecast'], inplace=True)

# %%
if 'Refreshing' in df4.columns:
    df4.drop(columns=['Refreshing'], inplace=True)

# %%
if 'Skin-Barrier' in df4.columns:
    df4.drop(columns=['Skin-Barrier'], inplace=True)

# %%
if 'Soothing' in df4.columns:
    df4.drop(columns=['Soothing'], inplace=True)

# %%
if 'price' in df5.columns:
    df5.drop(columns=['price'], inplace=True)

# %%
if 'description' in df4.columns:
    df4.drop(columns=['description'], inplace=True)

# %%
if 'notable_effects' in df5.columns:
    df5.drop(columns=['notable_effects'], inplace=True)

# %%
if 'skintype' in df5.columns:
    df5.drop(columns=['skintype'], inplace=True)

# %%
if 'labels' in df5.columns:
    df5.drop(columns=['labels'], inplace=True)

# %%
if 'Balancing' in df5.columns:
    df5.drop(columns=['Balancing'], inplace=True)

# %%
df5.isnull().sum()

# %%
df5 = df4.dropna(subset=['product_type'])


# %%
binary_cols = [
    'Sensitive', 'Combination', 'Oily', 'Dry', 'Normal',
    'Acne-Free', 'Anti-Aging', 'Black-Spot',
    'Brightening', 'Hydrating', 'Moisturizing', 'Oil-Control',
    'Pore-Care', 'UV-Protection'
]

df5[binary_cols] = df5[binary_cols].fillna(0)



# %%
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df5['product_type_encoded'] = label_encoder.fit_transform(df5['product_type'])
product_type_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# %%
df5.columns

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Sum notable effects per product type
effect_cols = [
    'Acne-Free', 'Anti-Aging', 'Black-Spot',
    'Brightening', 'Hydrating', 'Moisturizing',
    'Oil-Control', 'Pore-Care', 'UV-Protection'
]
grouped = df5.groupby('product_type')[effect_cols].sum()

# Plot stacked bar chart
grouped.T.plot(kind='bar', stacked=True, figsize=(14, 6))
plt.title("Notable Effects by Product Type")
plt.ylabel("Number of Products")
plt.xlabel("Notable Effect")
plt.legend(title="Product Type")
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Define notable effects and skin type columns
effect_cols = [
    'Acne-Free', 'Anti-Aging', 'Black-Spot',
    'Brightening', 'Hydrating', 'Moisturizing',
    'Oil-Control', 'Pore-Care', 'UV-Protection'
]

skin_type_cols = ['Sensitive', 'Combination', 'Oily', 'Dry', 'Normal']

# Combine both effect and skin type columns
all_effects = effect_cols + skin_type_cols

# Group by product_type and sum each effect/skin type column
grouped_all = df5.groupby('product_type')[all_effects].sum()

# Plot stacked bar chart
grouped_all.T.plot(kind='bar', stacked=True, figsize=(16, 7))
plt.title("Notable Effects and Skin Types by Product Type")
plt.ylabel("Number of Products")
plt.xlabel("Effect / Skin Type")
plt.legend(title="Product Type")
plt.tight_layout()
plt.show()


# %%
df5

# %%
df5['product_type'].value_counts().plot(kind='bar', title='Product Type Distribution')


# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Correctly list columns that exist
cols_to_corr = [
    'Sensitive', 'Combination', 'Oily', 'Dry', 'Normal',
    'Acne-Free', 'Anti-Aging', 'Black-Spot', 'Brightening',
    'Hydrating', 'Moisturizing', 'Oil-Control', 'Pore-Care', 'UV-Protection'
]

# Compute correlation matrix
corr_matrix = df[cols_to_corr].corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True)
plt.title("Correlation Between Skin Types and Notable Effects")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Define columns
binary_cols = [
    'sensitive', 'combination', 'oily', 'dry', 'normal',
    'acne_free', 'anti_aging', 'black_spot', 'brightening',
    'hydrating', 'moisturizing', 'oil_control', 'pore_care', 'uv_protection'
]

# Univariate Analysis: Bar plots for categorical/binary columns
for col in ['product_type'] + binary_cols:
    plt.figure(figsize=(6, 4))
    df[col].value_counts(dropna=False).plot(kind='bar')
    plt.title(f'Distribution of {col}')
    plt.ylabel("Count")
    plt.xlabel(col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Bivariate Analysis: Relationship between product_type and each effect
for col in binary_cols:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='product_type', hue=col)
    plt.title(f'{col} by Product Type')
    plt.ylabel("Count")
    plt.xlabel("Product Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# %%
# Re-import libraries and reload the dataset after reset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the previously uploaded cleaned dataset


# Fill missing values for plotting purposes
df5.fillna("Unknown", inplace=True)

# Define binary/multi-label columns
binary_cols = [
    'Sensitive', 'Combination', 'Oily', 'Dry', 'Normal',
    'Acne-Free', 'Anti-Aging', 'Black-Spot', 'Brightening',
    'Hydrating', 'Moisturizing', 'Oil-Control', 'Pore-Care', 'UV-Protection'
]

# Univariate Analysis: Distribution of individual columns
for col in ['product_type'] + binary_cols:
    plt.figure(figsize=(6, 4))
    df[col].value_counts(dropna=False).plot(kind='bar')
    plt.title(f'Distribution of {col}')
    plt.ylabel("Count")
    plt.xlabel(col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Bivariate Analysis: Effect vs Product Type
for col in binary_cols:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='product_type', hue=col)
    plt.title(f'{col} by Product Type')
    plt.ylabel("Count")
    plt.xlabel("Product Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# %%


# %% [markdown]
# ##### 

# %%
df5

# %%
if 'product_href' in df5.columns:
    df5.drop(columns=['product_href'], inplace=True)

# %%
if 'product_name' in df5.columns:
    df5.drop(columns=['product_name'], inplace=True)

# %%
if 'Acne-Spot' in df5.columns:
    df5.drop(columns=['Acne-Spot'], inplace=True)

# %%
if 'skintype_list' in df5.columns:
    df5.drop(columns=['skintype_list'], inplace=True)

# %%
print('Acne-Free' in df5.columns)  


# %%
df5.isnull().sum()

# %%
# Define binary/multi-label columns
binary_cols = [
    'Sensitive', 'Combination', 'Oily', 'Dry', 'Normal',
    'Acne-Free', 'Anti-Aging', 'Black-Spot', 'Brightening',
    'Hydrating', 'Moisturizing', 'Oil-Control', 'Pore-Care', 'UV-Protection'
]

# Normalize column names for safety (optional but helpful)
df5.columns = df5.columns.str.strip().str.lower().str.replace('-', '_')

# Update the column names list accordingly (use lowercase + underscores)
binary_cols = [
    'sensitive', 'combination', 'oily', 'dry', 'normal',
    'acne_free', 'anti_aging', 'black_spot', 'brightening',
    'hydrating', 'moisturizing', 'oil_control', 'pore_care', 'uv_protection'
]

# Univariate Analysis
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


# %%
correlation_matrix = df5.corr(numeric_only=True)

plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f")
plt.title("Correlation Matrix of All Numeric/Binary Columns")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# %%
df5

# %%



