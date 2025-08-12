import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


df1=pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\Skincare Recommendation\cleaned_skincare_data.csv')

print(df1.head())



import matplotlib.pyplot as plt
import seaborn as sns


#univariate


num_cols = ['wrinkles_severity', 'acne_severity', 'dark_circle_severity',
            'pigmentation', 'redness', 'dark_spots']

for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df1[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.show()

#bivariate for numericals
import seaborn as sns
import matplotlib.pyplot as plt

# Set style
sns.set(style="whitegrid")

# List of numerical features to analyze
num_features = ['wrinkles_severity', 'acne_severity', 'dark_circle_severity', 
                'pigmentation', 'redness', 'dark_spots']

# Loop through each feature and create a bar chart
for feature in num_features:
    plt.figure(figsize=(10, 5))
    sns.barplot(x='product_recommendation', y=feature, data=df1, errorbar='sd')
    plt.title(f'Mean {feature} per Product Recommendation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



import pandas as pd

# Load the sample structure of df1 to infer categorical one-hot columns
columns = ['wrinkles_severity', 'acne_severity', 'dark_circle_severity',
           'pigmentation', 'redness', 'dark_spots', 'product_recommendation',
           '_combination', '_dry', '_normal', '_oily', '_sensitive',
           '_brown', '_dark', '_fair', '_medium', '_olive']

# Convert the list into a DataFrame just to analyze structure
df1_sample = pd.DataFrame(columns=columns)

# Extract one-hot encoded columns (those starting with an underscore)
cat_cols = [col for col in df1_sample.columns if col.startswith('_')]
cat_cols



#bivariate for categorial

for col in cat_cols:
    plt.figure(figsize=(6, 3))
    sns.barplot(x=df1[col], y=df1['product_recommendation'].astype(str), estimator=lambda x: len(x), ci=None)
    plt.title(f'{col} vs Product Recommendation')
    plt.xlabel(f"{col} (0 = No, 1 = Yes)")
    plt.ylabel("Count of Product Recommendation")
    plt.xticks([0, 1])
    plt.tight_layout()
    plt.show()

