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
git clone https://github.com/yourusername/password-strength-classifier.git
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

### 2. Mount Google Drive (Optional)

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

This code mounts Google Drive if you're running the script on Google Colab. It's optional and can be skipped if not needed.

### 3. Load and Preprocess Data

```python
filepath = '/content/drive/MyDrive/data.csv'
data = pd.read_csv(filepath, delimiter=',', on_bad_lines='skip', engine='python')
data['password'].fillna('', inplace=True)
X = data['password']
y = data['strength']
X = X.apply(lambda x: x[:20])
```

This section loads the dataset, handles missing values, extracts the 'password' and 'strength' columns, and truncates passwords to 20 characters for feature extraction.

### 4. Feature Extraction

```python
vectorizer = TfidfVectorizer(analyzer='char', lowercase=False, max_features=500)
X_vectorized = vectorizer.fit_transform(X)
```

TfidfVectorizer is used to convert password strings into numerical vectors based on character frequencies.

### 5. Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
```

The dataset is split into training and testing sets to evaluate the model's performance.

### 6. Model Training

```python
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
```

A Logistic Regression model is trained on the training data.

### 7. Model Evaluation

```python
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
```

The accuracy of the trained model is evaluated on the test data.

### 8. Predictions for New Passwords

```python
new_passwords = ['1j%bsQ*<+Ynz','kucing123','##barbie123','ajd1348#28t**','ppppppppp','adfkilws','nanananannanannan','anissajulianty07','anissa','123456','abcdef']
new_passwords_vectorized = vectorizer.transform(new_passwords)
predictions = logreg.predict(new_passwords_vectorized)
print("Predictions for new passwords:")
for password, prediction in zip(new_passwords, predictions):
    print(f"{password}: {prediction}")
```

The trained model is used to predict the strength of new passwords provided in the script.

---
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
