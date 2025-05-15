# Bushfire Prediction

Bushfire Prediction leverages historical rainfall and fire incident attributes to forecast bushfire risk in Australia using a Multilayer Perceptron (MLP) model. :contentReference[oaicite:0]{index=0}

## Project Overview  
This repository contains a Python implementation of an `MLPClassifier` for binary classification of bushfire risk days, utilising merged rainfall and fire attribute datasets. :contentReference[oaicite:1]{index=1}

## Table of Contents  
- [Installation](#installation)  
- [Data Description](#data-description)  
- [Usage](#usage)  
- [Model Implementation](#model-implementation)  
- [Evaluation](#evaluation)  
- [Project Structure](#project-structure)  
- [Contributing](#contributing)  
- [Licence](#licence)  
- [Acknowledgements](#acknowledgements)  

## Installation  
Clone the repository and install dependencies:  
```bash
git clone https://github.com/ErrorChen/Bushfire-prediction.git    :contentReference[oaicite:2]{index=2}
cd Bushfire-prediction  
pip install -r requirements.txt    :contentReference[oaicite:3]{index=3}
```  

## Data Description  
The `datasets/` folder includes:  
- **rainfall.csv**: Daily rainfall measurements (2016–2021) for Australian stations, recording precipitation amounts in millimetres. :contentReference[oaicite:4]{index=4}  
- **fire_for16-21_attributes.csv**: Bushfire incident attributes from 2016 to 2021, detailing fire area, duration and location coordinates (latitude/longitude). :contentReference[oaicite:5]{index=5}  

## Usage  
Adjust the data path in `MLP.py` before running:  
```python
X, y = load_data('datasets/rainfall.csv')  # or 'datasets/fire_for16-21_attributes.csv' :contentReference[oaicite:6]{index=6}
```  
Then execute:  
```bash
python MLP.py    :contentReference[oaicite:7]{index=7}
```  
This will load the CSV, preprocess features, split into training/testing sets, train the `MLPClassifier`, and output the classification report.

## Model Implementation  
- **Data loading**: `load_data()` reads a CSV into a feature matrix `X` and label vector `y` using `pandas.read_csv()`. :contentReference[oaicite:8]{index=8}  
- **Classifier**:  
  ```python
  from sklearn.neural_network import MLPClassifier

  mlp = MLPClassifier(
      hidden_layer_sizes=(100, 50),
      activation='relu',
      solver='adam',
      alpha=1e-4,
      learning_rate_init=1e-3,
      max_iter=200,
      random_state=42
  )
  ```  
  These hyperparameters balance model capacity and training stability. :contentReference[oaicite:9]{index=9}

## Evaluation  
Performance is measured using `classification_report` from `sklearn.metrics`, reporting precision, recall, F1-score and support for each class. :contentReference[oaicite:10]{index=10}

## Project Structure  
```  
Bushfire-prediction/              :contentReference[oaicite:11]{index=11}
├── datasets/  
│   ├── fire_for16-21_attributes.csv  
│   └── rainfall.csv  
├── MLP.py                     # MLP training & evaluation script   :contentReference[oaicite:12]{index=12}
├── LSTM.py                    # Placeholder for LSTM model         :contentReference[oaicite:13]{index=13}
├── proj.code-workspace        # VS Code workspace settings         :contentReference[oaicite:14]{index=14}
├── requirements.txt           # Dependency list                    :contentReference[oaicite:15]{index=15}
└── LICENSE                    # MIT licence                        :contentReference[oaicite:16]{index=16}
```

## Contributing  
1. Fork the repository.  
2. Create a branch: `git checkout -b feature/your-feature`.  
3. Commit changes: `git commit -m "Add feature"`.  
4. Push: `git push origin feature/your-feature`.  
5. Open a Pull Request.

## Licence  
Released under the MIT Licence. See [LICENSE](LICENSE). :contentReference[oaicite:17]{index=17}

## Acknowledgements  
- Weather Australia dataset overview (Kaggle) :contentReference[oaicite:18]{index=18}  
- “Rain in Australia” rainfall details (CSDN) :contentReference[oaicite:19]{index=19}  
- scikit-learn’s `MLPClassifier` documentation :contentReference[oaicite:20]{index=20}  
- ENGG2112 course resources, University of Sydney
::contentReference[oaicite:21]{index=21}
