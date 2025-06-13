Awesome! Let's build a **Linear Regression** model together 🧠📈

---

## ✅ Project Explanation

You will be provided with step-by-step instructions on how to build the machine learning model, along with a Python notebook in **JupyterLab** where you can type and run the code.

To use the Jupyter Notebook:

* You can enter code by clicking on a **cell**. To open a new cell, hover in the middle of the notebook and click the **"+" icon** or use the **Insert menu**.
* Follow each step in the instructions and type the corresponding code into a cell.
* To run a cell, either click the **play button** on the left side of the cell or press **Shift + Enter** on your keyboard.

If you're unfamiliar with how Jupyter Notebooks work, you can search online for tutorials or guides on “How to use Jupyter Notebook” for more details.

---

## 🎯 Project: Build a Simple Linear Regression Model

---

### 🔢 Step 1: Install Required Libraries (if needed)

📌 This step is for making sure you have everything you need to run the code.

🧩
Copy the code below and paste it where it says `Paste Step 1 Code Here` then click the play button.
Feel free to skip the explanation below and go to the next step.

```python
# Step 1: Install scikit-learn (only run this if you're on a fresh system)
!pip install -U scikit-learn
```

🔎 **Explanation**:
`scikit-learn` is the library we'll use to build the linear regression model.

---

### 🧪 Step 2: Create the Training Data

📌 In this step, you'll manually enter your own data (like how many hours someone studied and the score they got). You can change the numbers if you want!

🧩
Copy the code below and paste it where it says `Paste Step 2 Code Here` then click the play button.
Feel free to skip the explanation below and go to the next step.

```python
# Step 2: Create training data
hours_studied = [1, 2, 3, 4, 5]  # You can change these numbers!
test_scores = [50, 55, 65, 70, 75]  # ← Replace these with your own scores!
```

🔎 **Explanation**:
We're making two lists: one for hours studied, and one for test scores. These are our training data.

---

### 🔧 Step 3: Reshape the Data (so the model can understand it)

🧩
Copy the code below and paste it where it says `Paste Step 3 Code Here` then click the play button.
Feel free to skip the explanation below and go to the next step.

```python
# Step 3: Reshape the data
import numpy as np

X = np.array(hours_studied).reshape(-1, 1)
y = np.array(test_scores)
```

🔎 **Explanation**:
We convert our Python lists into NumPy arrays and reshape the input so it fits what the model expects.

---

### 🧠 Step 4: Train the Linear Regression Model

🧩
Copy the code below and paste it where it says `Paste Step 4 Code Here` then click the play button.
Feel free to skip the explanation below and go to the next step.

```python
# Step 4: Train the linear regression model
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)
```

🔎 **Explanation**:
We’re telling the model to find a pattern in the data using `.fit()` — this is the training step!

---

### 📈 Step 5: Make a Prediction

📌 You can change the number of study hours to try different predictions!

🧩
Copy the code below and paste it where it says `Paste Step 5 Code Here` then click the play button.
Feel free to skip the explanation below and go to the next step.

```python
# Step 5: Make a prediction
hours = 6  # ← Change this to test other values!
predicted_score = model.predict([[hours]])
print(f"If you study for {hours} hours, your predicted score is {predicted_score[0]:.2f}")
```

🔎 **Explanation**:
We give the model a number (like 6 hours studied) and it predicts the test score based on the pattern it learned.

---

### 📊 Step 6: Visualize the Results (Optional but Fun!)

🧩
Copy the code below and paste it where it says `Paste Step 6 Code Here` then click the play button.
Feel free to skip the explanation below and go to the next step.

```python
# Step 6: Visualize the results
import matplotlib.pyplot as plt

plt.scatter(X, y, color='blue', label='Actual Scores')
plt.plot(X, model.predict(X), color='red', label='Prediction Line')
plt.xlabel('Hours Studied')
plt.ylabel('Test Score')
plt.title('Study Time vs Test Score')
plt.legend()
plt.grid(True)
plt.show()
```

🔎 **Explanation**:
This makes a graph showing your original data points (blue dots) and the prediction line (red line).

---

### 🎉 You're Done!

You’ve built a working linear regression model all by yourself! 💪✨
You can now try:

* Changing the `hours_studied` and `test_scores` values in Step 2.
* Predicting different hours in Step 5.
* Rerunning the notebook with new data!

Let me know if you'd like to try building another model like decision tree, classification, or even a deep learning neural net! 🧠💻