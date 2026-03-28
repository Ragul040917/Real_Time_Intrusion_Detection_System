import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from itertools import cycle
import numpy as np

# Load the dataset
data = pd.read_csv('traffic_prp_200.csv')

# Handle missing values if any
data = data.fillna(method='ffill')

# Encode categorical variables
label_encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = label_encoder.fit_transform(data[column])

# Define features and target variable
X = data.drop('label', axis=1)
y = data['label']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'SVM': SVC(probability=True),
    'Random Forest': RandomForestClassifier(),
    'Neural Network': MLPClassifier()
}

# Train and evaluate models
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    results[model_name] = {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, average='weighted'),
        'recall': recall_score(y_test, predictions, average='weighted'),
        'f1_score': f1_score(y_test, predictions, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, predictions),
        'roc_curve': roc_curve(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, "predict_proba") else None,
        'roc_auc': auc(*roc_curve(y_test, model.predict_proba(X_test)[:, 1])[:2]) if hasattr(model, "predict_proba") else None
    }
    print(f"{model_name} Classification Report:\n", classification_report(y_test, predictions))

# Visualization

# Accuracy, Precision, Recall, F1-Score Bar Chart
metrics_df = pd.DataFrame({
    'Model': results.keys(),
    'Accuracy': [result['accuracy'] for result in results.values()],
    'Precision': [result['precision'] for result in results.values()],
    'Recall': [result['recall'] for result in results.values()],
    'F1-Score': [result['f1_score'] for result in results.values()]
})

plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='value', hue='variable', data=pd.melt(metrics_df, id_vars=['Model']))
plt.title('Model Performance Metrics')
plt.ylabel('Score')
plt.show()

# Confusion Matrix
for model_name, result in results.items():
    plt.figure(figsize=(8, 6))
    sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# ROC Curves
plt.figure(figsize=(10, 8))
for model_name, result in results.items():
    if result['roc_curve']:
        fpr, tpr, _ = result['roc_curve']
        roc_auc = result['roc_auc']
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.show()
