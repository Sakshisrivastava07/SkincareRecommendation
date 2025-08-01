import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\Skincare Recommendation\final_skincare_dataset.csv')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# X = features, y = label
# X = df5.drop(columns=["product_type", "product_type_encoded"])
# y = df5["product_type_encoded"]
# #y = df5[["Facewash", "Toner", "Serum", "Moisturizer","Sunscreen"]]  # binary columns for each product


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# classes = np.unique(y_train)
# weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
# class_weights = dict(zip(classes, weights))

# rf = RandomForestClassifier(class_weight=class_weights, random_state=42)

# param_grid = {
#     'n_estimators': [100, 200,300],
#     'max_depth': [5,10, 20, None],
#     'min_samples_split': [2, 5,10],
#     'min_samples_leaf': [1, 2,4],
#     'max_features':['sqrt','log2']
# }

# # Grid Search with F1 weighted scoring
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
# grid_search.fit(X_train, y_train)

# # Evaluate
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test)

# print("Best Parameters:", grid_search.best_params_)
# print("\nClassification Report:\n", classification_report(y_test, y_pred))


# df = pd.read_csv("final_skincare_dataset.csv")


X = df.drop(columns=["product_type", "product_type_encoded"])
y = df["product_type_encoded"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier( random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],           # Try more trees
    'max_depth': [None, 20, 30, 50],           # Explore deeper trees
    'min_samples_split': [2, 5, 10],           # Controls when to split a node
    'min_samples_leaf': [1, 2, 4],             # Minimum samples per leaf node
    'max_features': ['sqrt', 'log2']           # Feature sampling for splits
}



grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Best Parameters:", grid_search.best_params_)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

