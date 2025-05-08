import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# G’day! Load the bushfire-prediction weather data CSV
df = pd.read_csv('AUSWeatherData.csv')

# Select the eight features we’ll use as inputs
# (e.g. MinTemp, MaxTemp, Rainfall, WindGustSpeed, WindSpeed9am,
#  Humidity9am, Pressure9am, Temp9am)
features = [
    'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed',
    'WindSpeed9am', 'Humidity9am', 'Pressure9am', 'Temp9am'
]
df = df[features + ['RainTomorrow']]

# Drop any rows with missing values
df.dropna(inplace=True)

# Encode the target: “No” → 0, “Yes” → 1
df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1})

# Split into features (X) and target (y)
X = df[features]
y = df['RainTomorrow']

# Split into training and test sets (80/20), with a fixed random state
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardise the inputs (zero mean, unit variance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Set up the MLP: two hidden layers of 32 neurons each, ReLU activation
clf = MLPClassifier(
    hidden_layer_sizes=(32, 32),
    activation='relu',
    solver='adam',
    max_iter=200,
    random_state=42
)

# Train the network
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Output performance metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Rain', 'Rain']))

