import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
dataset = pd.read_csv('customer_churn_large_dataset.csv')

#checks for missing values in dataset
missing_values = dataset.isna().sum()

print(missing_values)


# Calculate the correlation matrix so as to understand unnecessary features
correlation_matrix = dataset.corr()

# Plot a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Get the correlation values with respect to the target variable
correlation_with_target = correlation_matrix["Churn"].sort_values(ascending=False)
print(correlation_with_target)

# Drop columns based on correlation analysis
columns_to_drop = ["CustomerID", "Name", "Location"]
dataset.drop(columns=columns_to_drop, inplace=True)


# Define mapping for encoding gender
gender_mapping = {'Male': 0, 'Female': 1}

# Encode the "Gender" column
dataset['Gender'] = dataset['Gender'].map(gender_mapping)



# Separate features and target
X = dataset.drop("Churn", axis=1)
y = dataset["Churn"]



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree model
decision_tree_model = DecisionTreeClassifier(random_state=42)

# Train the model on the training data
decision_tree_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = decision_tree_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)

# Save the model using pickle
pickle.dump(decision_tree_model, open('model.pkl', 'wb'))





















































