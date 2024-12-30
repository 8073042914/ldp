import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


def main():
    st.title("Liver Disease Prediction App")

    # File Uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Data Preprocessing
        # Outlier Handling
        def replace_outliers_iqr(df, column):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

        for col in data.select_dtypes(include=['float64', 'int64']).columns:
            replace_outliers_iqr(data, col)

        # Feature Scaling
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data.select_dtypes(include=[np.number]))
        data_scaled = pd.DataFrame(scaled_features, columns=data.select_dtypes(include=[np.number]).columns)
        data_scaled['category'] = data['category']  # Add the target variable back

        # Model Building and Prediction
        x = data_scaled.drop("category", axis=1)
        y = data_scaled["category"]
        X = pd.get_dummies(x, drop_first=True)

        if X.isnull().sum().sum() > 0:
            X = X.fillna(X.mean())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Selection
        model_choice = st.selectbox("Choose a model:", ["Logistic Regression", "KNN", "SVM", "Naive Bayes", "Random Forest"])

        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_choice == "KNN":
            model = KNeighborsClassifier(n_neighbors=5)
        elif model_choice == "SVM":
            model = SVC(kernel='linear')
        elif model_choice == "Naive Bayes":
            model = GaussianNB()
        elif model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Display Results
        st.write("Prediction Results:")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Display Confusion Matrix
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

if __name__ == "__main__":
    main()