# Bushfire Prediction

Leveraging historical meteorological observations to forecast bushfire risk in Australia using machine learning models.

## Table of Contents

* [Project Overview](#project-overview)
* [Installation](#installation)
* [Data](#data)
* [Usage](#usage)
* [Model Implementations](#model-implementations)
* [Results](#results)
* [Project Structure](#project-structure)
* [Contributing](#contributing)
* [License](#license)
* [Acknowledgements](#acknowledgements)

## Project Overview

This repository provides Python implementations of two models to classify daily bushfire risk:

1. **Multilayer Perceptron (MLP)**: A fully connected neural network using `sklearn.neural_network.MLPClassifier`.
2. **Long Short-Term Memory (LSTM)**: A recurrent neural network for sequence modeling in `LSTM.py`.

Both models are trained on the [WeatherAUS dataset](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package) and output a binary risk label (High/Low). Scripts handle data preprocessing, model training, and evaluation.

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/ErrorChen/Bushfire-prediction.git
   cd Bushfire-prediction
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

Dependencies:

* `pandas`
* `numpy`
* `scikit-learn`
* `tensorflow` (for LSTM)

## Data

1. **Download** the WeatherAUS dataset from Kaggle:
   [https://www.kaggle.com/jsphyg/weather-dataset-rattle-package](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)

2. **Place** the unzipped `weatherAUS.csv` file into the `datasets/` directory.

The dataset includes daily weather observations (temperature, rainfall, humidity, wind speed, pressure) across multiple Australian stations.

## Usage

### MLP Model

Run the MLP classifier:

```bash
python MLP.py \
  --data-path datasets/weatherAUS.csv \
  --target-variable Risk  \
  --hidden-layers 100 50 \
  --epochs 200
```

### LSTM Model

Run the LSTM model for sequence prediction:

```bash
python LSTM.py \
  --data-path datasets/weatherAUS.csv \
  --sequence-length 10 \
  --batch-size 32 \
  --epochs 50
```

Scripts automatically:

* Load and clean the CSV
* Encode categorical features (one-hot)
* Impute missing values
* Split data into training/testing sets
* Train and evaluate the model

## Model Implementations

### MLPClassifier

* **Architecture**: Two hidden layers (`hidden_layer_sizes=(100, 50)`)
* **Activation**: ReLU
* **Solver**: Adam
* **Hyperparameters**:

  * `alpha=1e-4`
  * `learning_rate_init=1e-3`
  * `max_iter=200`
  * `random_state=42`

### LSTM

* **Architecture**: Single LSTM layer with 64 units, followed by Dense output
* **Loss**: Binary crossentropy
* **Optimizer**: Adam
* **Hyperparameters**:

  * `sequence_length`: length of input time window
  * `batch_size`: training batch size
  * `epochs`: number of training epochs

## Results

After training, both scripts print a classification report:

```
              precision    recall  f1-score   support

        Low       0.85      0.90      0.87     20000
       High       0.82      0.75      0.78     15000

   accuracy                           0.84     35000
  macro avg       0.84      0.83      0.83     35000
weighted avg       0.84      0.84      0.84     35000
```

Feel free to adjust hyperparameters to improve performance.

## Project Structure

```
Bushfire-prediction/
├── datasets/             # Raw and processed data files
│   └── weatherAUS.csv
├── MLP.py                # MLPClassifier training and evaluation
├── LSTM.py               # LSTM model training and evaluation
├── requirements.txt      # Python dependencies
├── proj.code-workspace   # VS Code workspace settings
├── LICENSE               # MIT license
└── README.md             # Project documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/...`)
3. Commit your changes (`git commit -m "Add ..."`)
4. Push to your branch (`git push origin feature/...`)
5. Open a Pull Request for review

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

* WeatherAUS dataset on Kaggle
* Scikit-learn and TensorFlow libraries
* University of Sydney ENGG2112 course materials
