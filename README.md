#  Spam Email Detection using Machine Learning

##  Project Overview

This project is a **Machine Learning-based Spam Email Detection system** that classifies emails as **Spam** or **Not Spam** using the **Gaussian Naive Bayes** algorithm.

The project uses the **Spambase dataset** and evaluates the trained model using accuracy, classification report, and confusion matrix.

---

##  Features

*  Loads and processes the Spambase dataset
*  Separates input features and target class
*  Splits data into training and testing sets
*  Uses Gaussian Naive Bayes for classification
*  Calculates model accuracy
*  Generates a classification report
* Generates a confusion matrix

---

##  Technologies Used

* **Python 3**
* **Pandas**
* **Scikit-learn**
* **Gaussian Naive Bayes**
* **Machine Learning**

---

##  Project Structure

```text
Spam-Email-Detector/
│
├── spam_email_detector.py
├── spambase_csv.csv
└── README.md
```

---

##  Dataset

The project uses the **Spambase dataset**.

The dataset contains multiple email-related features and a target column:

```text
class
```

The `class` column is used as the target variable, while the remaining columns are used as input features.

```python
X = df.drop("class", axis=1)
y = df["class"]
```

---

##  Machine Learning Algorithm

### Gaussian Naive Bayes

The project uses:

```python
model = GaussianNB()
```

Gaussian Naive Bayes is a probabilistic classification algorithm based on **Bayes' theorem**.

It assumes that the features follow a Gaussian (normal) distribution within each class.

### Why Naive Bayes?

* Fast training
* Simple implementation
* Works well for classification
* Suitable for high-dimensional feature sets
* Low computational cost

---

##  Machine Learning Workflow

```text
Dataset
   ↓
Load Data
   ↓
Separate Features & Target
   ↓
Train/Test Split
   ↓
Train Gaussian Naive Bayes
   ↓
Predict Test Data
   ↓
Evaluate Model
   ↓
Accuracy + Classification Report
   ↓
Confusion Matrix
```

---

##  Train-Test Split

The dataset is divided into:

```text
80% → Training Data
20% → Testing Data
```

The project uses:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

##  Model Evaluation

The project evaluates the model using three metrics/tools.

###  Accuracy

Measures the percentage of correctly classified emails.

```python
accuracy_score(y_test, pred)
```

---

###  Classification Report

The classification report provides:

* Precision
* Recall
* F1-score
* Support

```python
classification_report(y_test, pred)
```

---

### Confusion Matrix

The confusion matrix shows the number of:

* True Positives
* True Negatives
* False Positives
* False Negatives

```python
confusion_matrix(y_test, pred)
```

---

##  How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Spam-Email-Detector.git
```

### 2. Navigate to the Project

```bash
cd Spam-Email-Detector
```

### 3. Install Dependencies

```bash
pip install pandas scikit-learn
```

### 4. Run the Program

```bash
python spam_email_detector.py
```

---

##  Expected Output

The program displays:

```text
============================================================
SPAM EMAIL DETECTION
============================================================

[Dataset Preview]

Accuracy: XX.XX

Classification Report
              precision    recall    f1-score

...

Confusion Matrix
[[... ...]
 [... ...]]
```

---

##  Concepts Demonstrated

This project demonstrates practical understanding of:

* Python
* Pandas
* Data preprocessing
* Feature-target separation
* Train-test splitting
* Supervised Machine Learning
* Classification
* Gaussian Naive Bayes
* Model prediction
* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

##  Future Improvements

Possible improvements include:

* Add a user interface for entering email text
* Natural Language Processing (NLP)
* TF-IDF feature extraction
* Compare Naive Bayes with Logistic Regression
* Compare with Random Forest
* Add model visualization
* Build a Streamlit web application
* Save the trained model using Joblib
* Deploy the application online

---

##  Important Note

This is an **educational Machine Learning project** demonstrating spam classification using the Spambase dataset. It is not intended to be used as a production-grade email security system.

---

##  Author

Developed as a Machine Learning project to understand **classification algorithms, dataset handling, model evaluation, and spam detection** using Python and Scikit-learn.


