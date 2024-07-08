#SUPPORT VECTOR MACHINE(SVM)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
titanic_data = pd.read_csv('titanic.csv')

# Fill missing values
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
titanic_data.drop(columns=['Cabin'], inplace=True)
titanic_data.dropna(inplace=True)

# Convert categorical variables into dummy/indicator variables
titanic_data = pd.get_dummies(titanic_data, columns=['Sex'], drop_first=True)

# Select relevant features and target
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male']
target = 'Survived'

# Prepare the feature matrix (X) and the target vector (y)
X = titanic_data[features]
y = titanic_data[target]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create and train the SVM model
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)