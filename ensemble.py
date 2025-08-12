# import pandas as pd
# import numpy as np

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.metrics import classification_report, hamming_loss, accuracy_score



# df1=pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\Skincare Recommendation\cleaned_skincare_dataset.csv')

# # Sample synthetic data (structure similar to user's description)
# np.random.seed(42)
# n_samples = 100
# X = pd.DataFrame({
#     'wrinkles_severity': np.random.randint(0, 11, n_samples),
#     'acne_severity': np.random.uniform(0, 10, n_samples),
#     'dark_circle_severity': np.random.uniform(0, 10, n_samples),
#     'pigmentation': np.random.randint(0, 11, n_samples),
#     'redness': np.random.uniform(0, 10, n_samples),
#     '_dark': np.random.randint(0, 2, n_samples),
#     '_fair': np.random.randint(0, 2, n_samples),
#     '_medium': np.random.randint(0, 2, n_samples),
#     '_olive': np.random.randint(0, 2, n_samples)
# })

# # Sample multi-label targets
# labels = [['facewash'], ['serum'], ['moisturizer'], ['toner'], ['sunscreen']]
# y_raw = [list(np.random.choice([label[0] for label in labels], size=np.random.randint(1, 3), replace=False)) for _ in range(n_samples)]

# # Binarize labels
# mlb = MultiLabelBinarizer()
# y = mlb.fit_transform(y_raw)

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define base models
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
# lr = LogisticRegression(max_iter=1000)

# # Create an ensemble model using VotingClassifier wrapped in MultiOutputClassifier
# ensemble = MultiOutputClassifier(VotingClassifier(
#     estimators=[
#         ('rf', rf),
#         ('xgb', xgb),
#         ('lr', lr)
#     ],
#     voting='soft'
# ))

# # Train ensemble
# ensemble.fit(X_train, y_train)
# y_pred = ensemble.predict(X_test)

# # Evaluate
# report = classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=0)
# hamming = hamming_loss(y_test, y_pred)
# subset_acc = accuracy_score(y_test, y_pred)

# (report, hamming, subset_acc)


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, hamming_loss, accuracy_score
import warnings
warnings.filterwarnings("ignore")  # Suppress unnecessary warnings

# Load the cleaned skincare dataset
df1 = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\Skincare Recommendation\cleaned_skincare_dataset.csv')

# Ensure product_recommendation_clean is in list form
df1['product_recommendation_clean'] = df1['product_recommendation_clean'].apply(eval)

# Features (drop target)
X = df1.drop(columns=['product_recommendation_clean'])

# Target (multi-label)
y_raw = df1['product_recommendation_clean']

# Binarize the target
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y_raw)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define base models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
lr = LogisticRegression(max_iter=1000)

# Ensemble with VotingClassifier wrapped in MultiOutputClassifier
ensemble = MultiOutputClassifier(VotingClassifier(
    estimators=[
        ('rf', rf),
        ('xgb', xgb),
        ('lr', lr)
    ],
    voting='soft'
))

# Train
ensemble.fit(X_train, y_train)

# Predict
y_pred = ensemble.predict(X_test)

# Evaluate
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=0))
print("Hamming Loss:", hamming_loss(y_test, y_pred))
print("Subset Accuracy:", accuracy_score(y_test, y_pred))

