#LINEAR REGRESSION USING LINEAR ALGEBRA
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
titanic_data = pd.read_csv('titanic.csv')

# Fill missing values
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
titanic_data.drop(columns=['Cabin'], inplace=True)
titanic_data.dropna(inplace=True)

# Convert categorical variables into dummy/indicator variables
titanic_data = pd.get_dummies(titanic_data, columns=['Sex'], drop_first=True)

# Update the features list to include the correct dummy variable column name
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male']
target = 'Survived'

# Prepare the feature matrix (X) and the target vector (y)
X = titanic_data[features].values.astype(float)  # Ensure X is float64
y = titanic_data[target].values.astype(float)   # Ensure y is float64

# Add intercept term to X
intercept = np.ones((X.shape[0], 1))
X = np.concatenate((intercept, X), axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate parameters using normal equation (linear algebra approach)
theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# Make predictions on the test set
y_pred = X_test @ theta

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Display the model's coefficients (excluding intercept)
coefficients = pd.DataFrame(theta[1:], index=features, columns=['Coefficient'])
print(coefficients)

# Display the intercept (theta[0] is the intercept)
print(f"Intercept: {theta[0]}")