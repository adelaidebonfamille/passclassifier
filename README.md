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

### 3. Data Preparation

Prepare your dataset in CSV format. The dataset should contain two columns: 'password' and 'strength', where 'password' is the password string and 'strength' is the corresponding label indicating the strength of the password (e.g., 'weak', 'medium', 'strong').

### 4. Update Filepath

Update the `filepath` variable in the script to point to your dataset:

```python
filepath = '/path/to/your/data.csv'
```

### 5. Run the Script

Run the script using Python:

```bash
python password_strength_classifier.py
```

This will train the model on your dataset and output the accuracy of the model on the test data, as well as predictions for a list of new passwords provided in the script.

## Example

```python
python password_strength_classifier.py
```

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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to customize this README according to your project's specifics. Let me know if you need further assistance!
