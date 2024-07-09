#KNN
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('titanic.csv')

# Handle missing values for numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Handle missing values for non-numeric columns
non_numeric_cols = df.select_dtypes(include=['object']).columns
df[non_numeric_cols] = df[non_numeric_cols].fillna(df[non_numeric_cols].mode().iloc[0])

# Convert categorical columns to numerical using one-hot encoding
df = pd.get_dummies(df)

# Verify no missing values remain
if df.isnull().sum().any():
    raise ValueError("There are still missing values in the dataset.")

# Define features (X) and target (y)
# Assuming 'Survived' is the target variable
if 'Survived' in df.columns:
    X = df.drop('Survived', axis=1)
    y = df['Survived']
else:
    raise ValueError("The 'Survived' column is not found in the dataset")

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the k-NN classifier with k=5 (or any other value you choose)
knn = KNeighborsClassifier(n_neighbors=5)

# Train the classifier
knn.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test_scaled)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')