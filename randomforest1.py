import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


df1=pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\Skincare Recommendation\cleaned_skincare_dataset.csv')

import pandas as pd
import ast
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import classification_report, hamming_loss, accuracy_score

# --- Step 1: Load cleaned dataset


# --- Step 2: Convert product_recommendation_clean to list
def safe_convert(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return [x.strip()]
    return x

df1['product_recommendation_clean'] = df1['product_recommendation_clean'].apply(safe_convert)

# --- Step 3: Multi-label Binarization
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df1['product_recommendation_clean'])
label_names = mlb.classes_

# --- Step 4: Feature Preparation
X = df1.drop(['product_recommendation_clean'], axis=1)
X = X.select_dtypes(include=['int64', 'float64', 'uint8'])  # Keep only numerical & one-hot

# --- Step 5: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Step 6: Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Step 7: Grid Search for Random Forest
param_grid = {
    'estimator__n_estimators': [100, 200],
    'estimator__max_depth': [10, None],
    'estimator__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    MultiOutputClassifier(RandomForestClassifier(random_state=42)),
    param_grid,
    scoring='f1_micro',
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# --- Step 8: Prediction & Evaluation
y_pred = best_model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))

print("\nHamming Loss:", hamming_loss(y_test, y_pred))
print("Subset Accuracy:", accuracy_score(y_test, y_pred))
