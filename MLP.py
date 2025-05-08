import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# — Load dataset from CSV file —
# Assumes CSV contains columns: temperature, precipitation, humidity, wind_speed, ndvi, soil_moisture, slope, dmc, fire
df = pd.read_csv('bushfire_data.csv')

# — Select features and target —
X = df[['temperature', 'precipitation', 'humidity', 'wind_speed',
        'ndvi', 'soil_moisture', 'slope', 'dmc']].values
y = df['fire'].values

# — Split into training and test sets (80% train, 20% test) —
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# — Standardise features to accelerate convergence —
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# — Define MLP classifier with two hidden layers —
clf = MLPClassifier(
    hidden_layer_sizes=(16, 8),   # First hidden layer: 16 neurons; second: 8 neurons
    activation='relu',            # ReLU activation for hidden units
    solver='adam',                # Adam optimiser for weight updates
    max_iter=200,                 # Maximum of 200 training epochs
    random_state=42               # Ensure reproducibility
)

# — Train the perceptron on training data —
clf.fit(X_train_scaled, y_train)

# — Predict on test data and evaluate performance —
y_pred = clf.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# — Display sample prediction probabilities —
probs = clf.predict_proba(X_test_scaled)[:, 1]
for i in range(5):
    print(f"Sample {i}: Actual={y_test[i]}, Predicted probability of fire={probs[i]:.3f}")
