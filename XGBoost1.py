import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, hamming_loss, accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
import ast


df1=pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\Skincare Recommendation\cleaned_skincare_dataset.csv')


# Assuming df1 is already loaded and cleaned
# Make sure 'product_recommendation_clean' is a list (not a string)
# 

import pandas as pd
import ast
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, hamming_loss, accuracy_score
from xgboost import XGBClassifier

# --- Load your data


# --- Ensure product_recommendation_clean is in list format
df1['product_recommendation_clean'] = df1['product_recommendation_clean'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# --- Convert multi-label column to binary format
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df1['product_recommendation_clean'])
label_names = mlb.classes_

# --- Prepare features
X = df1.drop(['product_recommendation_clean'], axis=1)
X = X.select_dtypes(include=['int64', 'float64', 'uint8'])

# --- Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# --- Define base model and parameter grid
xgb = XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')

param_grid = {
    'estimator__n_estimators': [100, 150],
    'estimator__max_depth': [3, 5, 7],
    'estimator__learning_rate': [0.05, 0.1, 0.2],
    'estimator__subsample': [0.8, 1.0],
    'estimator__colsample_bytree': [0.8, 1.0]
}

# --- Wrap with MultiOutputClassifier
multi_xgb = MultiOutputClassifier(xgb)

# --- GridSearchCV
grid_search = GridSearchCV(
    estimator=multi_xgb,
    param_grid=param_grid,
    scoring='f1_weighted',
    cv=3,
    verbose=2,
    n_jobs=-1
)

# --- Fit the model
grid_search.fit(X_train, y_train)

# --- Best Parameters
print("Best parameters found: ", grid_search.best_params_)

# --- Evaluation
y_pred = grid_search.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))

# --- Additional Metrics
print("Hamming Loss:", hamming_loss(y_test, y_pred))
print("Subset Accuracy:", accuracy_score(y_test, y_pred))



import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from xgboost import XGBClassifier

# 🔹 Load your dataset
df = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\Skincare Recommendation\cleaned_skincare_dataset.csv')

# 🔹 Define features (X) and labels (y)
X = df.drop(columns=['product_recommendation_clean'])  # Change 'recommended_products' to your label column
y_raw = df['product_recommendation_clean'].apply(eval)  # If stored as strings like "['facewash', 'serum']"

# 🔹 Encode labels using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y_raw)

# 🔹 Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train MultiOutput XGBoost
multi_xgb = MultiOutputClassifier(XGBClassifier(eval_metric='logloss'))
multi_xgb.fit(X_train, y_train)

# 🔹 Save the model and the binarizer
joblib.dump(multi_xgb, 'xgb_model.pkl')
joblib.dump(mlb, 'mlb.pkl')


# import joblib

# # Save model
# joblib.dump(xgb, 'xgb_model.pkl')
# import joblib

# # Load model
# xgb_loaded = joblib.load('xgb_model.pkl')

# # Predict using loaded model
# y_pred = xgb_loaded.predict(X_test)


from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
import joblib

# Train
multi_xgb = MultiOutputClassifier(XGBClassifier(eval_metric='logloss'))
multi_xgb.fit(X_train, y_train)

# Save using joblib
joblib.dump(multi_xgb, 'xgb_model.pkl')

import joblib

# Load the trained model
xgb_loaded = joblib.load('xgb_model.pkl')

# Predict
y_pred = xgb_loaded.predict(X_test)



import joblib

model = joblib.load('xgb_model.pkl')
# print(type(model))
# print(model)



# import pandas as pd
# import joblib

# # 🔹 Load the saved model and MultiLabelBinarizer
# import pandas as pd
# import joblib

# # 🔹 Load the trained model and label binarizer
# model = joblib.load("xgb_model.pkl")   # Trained MultiOutputClassifier with XGBClassifier
# mlb = joblib.load("mlb.pkl")           # Trained MultiLabelBinarizer

# # 🔹 Define a new input (replace values based on user input or test cases)
# new_input = pd.DataFrame([{
#     'wrinkles_severity': 4,
#     'acne_severity': 7.5,
#     'dark_circle_severity': 6.0,
#     'pigmentation': 8,
#     'redness': 3.5,
#     '_dark': 0,
#     '_fair': 0,
#     '_medium': 1,
#     '_olive': 0
# }])

# # 🔹 Predict using the loaded model
# prediction = model.predict(new_input)

# # 🔹 Decode the prediction into product labels
# predicted_products = mlb.inverse_transform(prediction)

# # 🔹 Show the recommendation
# print(" Recommended Skincare Products:")
# print(predicted_products[0])

import pandas as pd
import joblib

# 🔹 Step 1: Load the saved model and label binarizer
model = joblib.load("xgb_model.pkl")
mlb = joblib.load("mlb.pkl")

# 🔹 Step 2: Provide new input data
new_input = pd.DataFrame([{
    'wrinkles_severity': 4,
    'acne_severity': 5.5,
    'dark_circle_severity': 3.0,
    'pigmentation': 9,
    'redness': 8.0,
    'dark_spots': 1,
    '_combination': 0,
    '_dry': 1,
    '_normal': 0,
    '_oily': 0,
    '_sensitive': 1,
    '_brown': 0,
    '_dark': 1,
    '_fair': 0,
    '_medium': 0,
    '_olive': 0,
}])


# #  Step 3: Predict probabilities
# probs = model.predict_proba(new_input)

# #  Step 4: Apply custom threshold
# custom_threshold = 0.3
# custom_pred = [[1 if prob[1] >= custom_threshold else 0 for prob in col] for col in zip(*probs)]

# # Step 5: Convert to product labels




# predicted_products = mlb.inverse_transform(custom_pred)



# # 🔹 Step 3: Predict and decode the recommended products
# prediction = model.predict(new_input)
# predicted_products = mlb.inverse_transform(prediction)

probs = model.predict_proba(new_input)

# 🔹 Step 4: Apply custom threshold
custom_threshold = 0.3
custom_pred = [[1 if prob[1] >= custom_threshold else 0 for prob in col] for col in zip(*probs)]

# ✅ Step 5: Convert to product labels
import numpy as np
custom_pred = np.array(custom_pred).reshape(1, -1)  # Ensure 2D shape for inverse_transform

predicted_products = mlb.inverse_transform(custom_pred)


# 🔹 Step 4: Show results
print("🧴 Recommended Skincare Products:")
print(predicted_products[0])

probs = model.predict_proba(new_input)

for i, label in enumerate(mlb.classes_):
    print(f"{label}: {probs[i][0][1]:.2f}")  # probability of label=1


import joblib

feature_order = X_train.columns.tolist()
joblib.dump(feature_order, "feature_order.pkl")
# also save model and mlb
joblib.dump(multi_xgb, "xgb_model.pkl")
joblib.dump(mlb, "mlb.pkl")


