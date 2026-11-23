# Experiment 3: Regression Techniques in Business Data(Simple Linear Regression)

##  Aim
To implement Simple Linear Regression in Python to study and predict the relationship between two business-related variables.

## Dataset
A sample business dataset representing the relationship between **Advertising Expenditure** (in Lakhs) and **Sales Revenue** (in Crores). The goal is to analyze how changes in advertising investment affect sales performance.

| Index | Advertising (Lakhs) | Sales (Crores) |
| :---: | :---: | :---: |
| 0 | 18 | 12 |
| 1 | 22 | 18 |
| 2 | 25 | 20 |
| 3 | 30 | 24 |
| 4 | 35 | 28 |
| 5 | 40 | 35 |
| 6 | 45 | 38 |
| 7 | 50 | 40 |
| 8 | 55 | 42 |
| 9 | 60 | 45 |

## Algorithm / Steps Followed
1. Import required Python libraries (`numpy`, `pandas`, `matplotlib`, `scikit-learn`).
2. Create and load the dataset into a Pandas DataFrame.
3. Display the dataset.
4. Select independent ($X$) and dependent ($Y$) variables and reshape $X$ into a 2D array.
5. Build a `LinearRegression` model using `sklearn.linear_model`.
6. Fit the model and calculate the intercept ($a$) and slope ($b$).
7. Predict $Y$ values for given $X$ values and evaluate performance using the $R^2$ score.
8. Plot the actual data points alongside the fitted regression line and predict sales for a new advertising budget.

##  Functions and Methods Used

| Category | Function / Method | Purpose / Description |
| :--- | :--- | :--- |
| **Data Creation / Handling** | `np.array()` | Create numerical arrays for regression input data |
| | `pd.DataFrame()` | Create or manage tabular data for regression |
| **Model Building (Scikit-learn)** | `LinearRegression()` | Initialize the Simple Linear Regression model |
| | `model.fit()` | Train the linear regression model using $X$ and $Y$ values |
| | `model.predict()` | Predict $Y$ values for given $X$ inputs using the fitted model |
| | `model.score()` | Calculate model performance ($R^2$ score) |
| **Visualization (Matplotlib)** | `plt.scatter()` | Plot scatter points (actual data points) |
| | `plt.plot()` | Plot regression line (predicted relationship) |
| | `plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `plt.show()` | Configure and display the regression plot |

##  Program & Output

# Step 1: Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 2: Create Dataset (Example Data Points)
# Example: Predicting Sales based on Advertising Budget

data = {
    'Advertising': [18, 22, 25, 30, 35, 40, 45, 50, 55, 60],
    'Sales': [12, 18, 20, 24, 28, 35, 38, 40, 42, 45]
}

# Create DataFrame
df = pd.DataFrame(data)
print("----- Dataset -----")
print(df)

# Step 3: Prepare Input (X) and Output (Y) Variables
X = np.array(df['Advertising']).reshape(-1, 1)
Y = np.array(df['Sales'])

# Step 4: Build and Train the Linear Regression Model
model = LinearRegression()
model.fit(X, Y)

# Step 5: Display Model Coefficients
print("\nIntercept (a):", model.intercept_)
print("Slope (b):", model.coef_[0])

# Regression Equation: Y = a + bX
print(f"\nRegression Equation: Y = {model.intercept_:.2f} + {model.coef_[0]:.2f}X")

# Step 6: Predict Y for Given X Values
Y_pred = model.predict(X)
print("\nPredicted Sales Values:")
print(Y_pred)

# Step 7: Evaluate Model Performance
r2 = model.score(X, Y)
print("\nModel Performance:")
print(f"R² Score = {r2:.4f}")

# Step 8: Visualize Regression Line
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color='blue', label='Actual Data Points')
plt.plot(X, Y_pred, color='red', linewidth=2, label='Regression Line')
plt.title("Simple Linear Regression: Advertising vs Sales")
plt.xlabel("Advertising Budget (in ₹ Lakhs)")
plt.ylabel("Sales (in ₹ Crores)")
plt.legend()
plt.show()

# Step 9: Example Prediction
# Predict Sales for a new Advertising value
new_value = np.array([[45]])
predicted_sales = model.predict(new_value)
print(f"\nPredicted Sales for Advertising ₹45 Lakhs = {predicted_sales[0]:.2f} Crores")

# Output

<img width="1920" height="1209" alt="image" src="https://github.com/user-attachments/assets/47a280c3-f4bd-42a9-b6d1-5b2348109b27" />

##  Result

Thus, Simple Linear Regression was successfully implemented using Python. The relationship between advertising expenditure and sales revenue was analysed, the regression equation was obtained, and future sales were predicted. The model achieved an R² score of 0.9736, indicating a strong positive relationship between advertising expenditure and sales revenue.

