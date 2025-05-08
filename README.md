# Bushfire Prediction

Bushfire Prediction leverages historical meteorological observations to forecast bushfire risk in Australia using a Multilayer Perceptron (MLP) model.

## Project Overview  
This repository contains a Python implementation of an `MLPClassifier` for binary classification of bushfire risk days based on the WeatherAUS dataset. All code is written in Python and utilises `pandas`, `numpy` and `scikit-learn` for data handling, model training and evaluation.

## Table of Contents  
- [Installation](#installation)  
- [Data Description](#data-description)  
- [Usage](#usage)  
- [Model Implementation](#model-implementation)  
- [Evaluation](#evaluation)  
- [Project Structure](#project-structure)  
- [Contributing](#contributing)  
- [Licence](#licence)  
- [Contact](#contact)  
- [Acknowledgements](#acknowledgements)  

## Installation  
Clone the repository and install the required packages:  
```bash
git clone https://github.com/ErrorChen/Bushfire-prediction.git
cd Bushfire-prediction
pip install pandas numpy scikit-learn
```

## Data Description  
The dataset (`AUSWeatherData.csv`) is sourced from the WeatherAUS collection on Kaggle, which provides daily weather observations from multiple Australian stations, including temperature, rainfall, humidity, wind speed and atmospheric pressure. It comprises approximately 145 000 rows and 23 columns, with a binary target indicating high or low bushfire risk.

## Usage  
Run the MLP training and evaluation script as follows:  
```bash
python MLP.py
```  
By default, the script loads the CSV, preprocesses features, splits into train/test sets, trains the MLPClassifier, and prints a classification report.

## Model Implementation  
- **Algorithm**: `sklearn.neural_network.MLPClassifier` with ReLU activation and Adam optimiser.  
- **Hyperparameters**:  
  - `hidden_layer_sizes=(100, 50)`  
  - `activation='relu'`  
  - `solver='adam'`  
  - `alpha=1e-4`  
  - `learning_rate_init=1e-3`  
  - `max_iter=200`  
  - `random_state=42`  

These settings provide a balance between capacity and training stability.

## Evaluation  
Model performance is assessed using the `classification_report` from `sklearn.metrics`, which outputs precision, recall, F1-score and support for each class. Example output is printed to the console upon script completion.

## Project Structure  
```
Bushfire-prediction/
├── AUSWeatherData.csv        # WeatherAUS dataset CSV
├── MLP.py                    # MLP model training & evaluation script
├── proj.code-workspace       # VS Code workspace settings
├── LICENSE                   # MIT licence
└── README.md                 # Project documentation
```

## Contributing  
1. Fork the repository.  
2. Create a branch: `git checkout -b feature/your-feature`.  
3. Commit your changes: `git commit -m "Add feature"`.  
4. Push to your branch: `git push origin feature/your-feature`.  
5. Open a Pull Request for review.

## Licence  
This project is released under the MIT Licence. See the [LICENSE](LICENSE) file for details.

## Contact  
ErrorChen (Group 23, USYD ENGG2112)  
Email: nicholas.tse@sydney.edu.au

## Acknowledgements  
- WeatherAUS dataset on Kaggle  
- Scikit-learn machine learning library  
- University of Sydney ENGG2112 course resources  
