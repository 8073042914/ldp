import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

# Load the data
data = pd.read_excel('project-data (1) (1).xlsx')  # Update file path if needed

# Data Preprocessing

# Handle missing values (replace with 0)
data.fillna(0, inplace=True)

# Outlier handling (using IQR method)
def replace_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

for col in data.select_dtypes(include=['float64', 'int64']).columns:
    replace_outliers_iqr(data, col)

# Encoding categorical features
sex_label_encoder = LabelEncoder()
category_label_encoder = LabelEncoder()
data['sex'] = sex_label_encoder.fit_transform(data['sex'])
data['category'] = category_label_encoder.fit_transform(data['category'])

# Feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.select_dtypes(include=[np.number]))
data_scaled = pd.DataFrame(scaled_features, columns=data.select_dtypes(include=[np.number]).columns)

# Streamlit UI elements
st.title("Liver Disease Prediction App")

# Display data information
st.write("Data Shape:", data.shape)
st.write("Data Head:")
st.dataframe(data.head())

# Model selection
model_choice = st.selectbox("Select a Model:", ["Logistic Regression", "Random Forest", "KNN", "SVM", "Gradient Boosting"])

# Model training and evaluation
x = data.drop("category", axis=1)
y = data["category"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_choice == "Random Forest":
    model = RandomForestClassifier()
elif model_choice == "KNN":
    model = KNeighborsClassifier(n_neighbors=5)
elif model_choice == "SVM":
    model = SVC()
elif model_choice == "Gradient Boosting":
    model = GradientBoostingClassifier()

model.fit(X_train, y_train)
y_prediction = model.predict(X_test)

# Display results
st.write(f"{model_choice} Results:")
st.write(f"Accuracy: {accuracy_score(y_test, y_prediction):.4f}")
st.write("Classification Report:\n", classification_report(y_test, y_prediction))
st.write("Confusion Matrix:\n", confusion_matrix(y_test, y_prediction))

# Feature importance visualization (for Random Forest)
if model_choice == "Random Forest":
    feature_importances = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False)
    st.write("Feature Importance:")
    st.bar_chart(feature_importances)  # Using Streamlit's bar_chart