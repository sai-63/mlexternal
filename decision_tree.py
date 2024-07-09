#DECISION TREE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('titanic.csv')

# Handle missing values
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Handle missing values for non-numeric columns
non_numeric_cols = df.select_dtypes(include=['object']).columns
df[non_numeric_cols] = df[non_numeric_cols].fillna(df[non_numeric_cols].mode().iloc[0])
df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))  # Filling categorical columns with mode

# Convert categorical columns to numerical using LabelEncoder
le = LabelEncoder()
df = df.apply(le.fit_transform)

# Define features (X) and target (y)
# Assuming 'Survived' is the target variable
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree classifier
clf = DecisionTreeClassifier(criterion='entropy')  # 'entropy' for ID3-like behavior

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')