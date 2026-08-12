import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("C:\\Users\\lenovo\\projects\\spam mail dectector\\spambase_csv.csv")

print("="*60)
print("SPAM EMAIL DETECTION")
print("="*60)

print(df.head())

# -----------------------
# Features and Target
# -----------------------
X = df.drop("class", axis=1)
y = df["class"]

# -----------------------
# Split Data
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------
# Train Model
# -----------------------
model = GaussianNB()

model.fit(X_train, y_train)

# -----------------------
# Prediction
# -----------------------
pred = model.predict(X_test)

# -----------------------
# Accuracy
# -----------------------
print("\nAccuracy:", accuracy_score(y_test, pred) * 100)

print("\nClassification Report")
print(classification_report(y_test, pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, pred))