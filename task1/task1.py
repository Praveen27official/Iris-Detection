# ===============================
# STEP 1: Import Libraries
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# STEP 2: Load Dataset
# ===============================
df = pd.read_csv("Iris.csv")

# ===============================
# STEP 3: Explore Data
# ===============================
print("First 5 rows:\n", df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nUnique Species:", df["Species"].unique())

# ===============================
# STEP 4: Data Visualization
# ===============================
sns.pairplot(df, hue="Species")
plt.show()

# Correlation heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.drop("Species", axis=1).corr(), annot=True)
plt.title("Feature Correlation")
plt.show()

# ===============================
# STEP 5: Prepare Data
# ===============================
X = df.drop("Species", axis=1)
y = df["Species"]

# ===============================
# STEP 6: Train-Test Split
# ===============================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# STEP 7: Train Model (KNN)
# ===============================
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# ===============================
# STEP 8: Evaluate Model
# ===============================
accuracy = model.score(X_test, y_test)
print("\nModel Accuracy:", accuracy)

# Detailed evaluation
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ===============================
# STEP 9: Predict New Data
# ===============================
new_data = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(new_data)

print("\nPrediction for new flower:", prediction)