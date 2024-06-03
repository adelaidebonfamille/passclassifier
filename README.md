# Password Strength Classifier

This Python script is designed to classify the strength of passwords using Logistic Regression. It takes a dataset containing passwords and their corresponding strength labels, preprocesses the data, trains a Logistic Regression model, and then predicts the strength of new passwords.

## Usage

### 1. Install Dependencies

Make sure you have the necessary dependencies installed. You can install them using pip:

```bash
pip install pandas scikit-learn
```

### 2. Clone Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/adelaidebonfamille/password-strength-classifier.git
```

### 3. Update Filepath

Update the `filepath` variable in the script to point to your dataset:

```python
filepath = '/path/to/your/data.csv'
```

### 5. Run the Script
This will train the model on your dataset and output the accuracy of the model on the test data, as well as predictions for a list of new passwords provided in the script.

## Code Explanation

### 1. Import Libraries

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

These libraries are imported for data manipulation, feature extraction, model training, and evaluation.

### 2. PasswordStrengthClassifier Class

```python
class PasswordStrengthClassifier:
    def __init__(self, filepath):
        # Constructor to initialize variables
    def load_data(self):
        # Method to load data from CSV file
    def preprocess_data(self):
        # Method to preprocess loaded data
    def train_model(self):
        # Method to train the Logistic Regression model
    def evaluate_model(self):
        # Method to evaluate the trained model
    def predict_password_strength(self, new_passwords):
        # Method to predict the strength of new passwords
```

The `PasswordStrengthClassifier` class encapsulates methods for loading data, preprocessing data, training the model, evaluating the model, and predicting password strengths.

### 3. Load Data

```python
filepath = input("Enter the path to the CSV file: ")
password_classifier = PasswordStrengthClassifier(filepath)
password_classifier.load_data()
```

The user is prompted to enter the path to the CSV file containing password data. The `load_data()` method reads the data from the file.

### 4. Preprocess Data

```python
password_classifier.preprocess_data()
```

The `preprocess_data()` method handles missing values, extracts features, and splits the data into training and testing sets.

### 5. Train Model

```python
password_classifier.train_model()
```

The `train_model()` method trains a Logistic Regression model on the training data.

### 6. Evaluate Model

```python
password_classifier.evaluate_model()
```

The `evaluate_model()` method evaluates the trained model's accuracy on the testing data.

### 7. Predict Password Strength

```python
new_passwords = ['1j%bsQ*<+Ynz','kucing123','##barbie123','ajd1348#28t**','ppppppppp','adfkilws','nanananannanannan','anissajulianty07','anissa','123456','abcdef']
password_classifier.predict_password_strength(new_passwords)
```

The `predict_password_strength()` method predicts the strength of new passwords provided in the script.

## Example Output

Output:
```
Model Accuracy: 0.85
Predictions for new passwords:
1j%bsQ*<+Ynz: strong
kucing123: weak
##barbie123: medium
ajd1348#28t**: strong
ppppppppp: weak
adfkilws: weak
nanananannanannan: medium
anissajulianty07: medium
anissa: weak
123456: weak
abcdef: weak
```
