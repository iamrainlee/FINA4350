import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load merged dataset into a Pandas dataframe
df = pd.read_csv('rate_FOMC.csv')

# Split data into features (X) and target (y)
X = df.drop(['Date', 'From_Date', 'data'], axis=1)
y = df['rate_hike']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Gaussian Naive Bayes model
nb = GaussianNB()

# Train the Naive Bayes model on the training set
nb.fit(X_train, y_train)

# Generate predictions on the testing set
y_pred = nb.predict(X_test)

# Evaluate the performance of the model using accuracy
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy)
