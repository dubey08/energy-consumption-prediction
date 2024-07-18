# Energy Consumption Prediction using LSTM and RNN

This repository contains the code and resources for predicting energy consumption using Long Short-Term Memory (LSTM) and Recurrent Neural Networks (RNN).

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to predict future energy consumption based on historical data using machine learning techniques. We utilize LSTM and RNN models, which are well-suited for time series prediction tasks due to their ability to capture temporal dependencies.

## Project Structure

├── data
│ ├── raw
│ ├── processed
├── notebooks
│ └── exploratory_data_analysis.ipynb
├── models
│ ├── lstm_model.py
│ ├── rnn_model.py
├── results
│ └── model_evaluation.png
├── src
│ ├── data_preprocessing.py
│ ├── model_training.py
│ ├── prediction.py
├── README.md
├── requirements.txt
└── LICENSE

- `data/`: Contains raw and processed data files.
- `notebooks/`: Jupyter notebooks for data exploration and analysis.
- `models/`: Scripts defining the LSTM and RNN models.
- `results/`: Directory to save model evaluation results and plots.
- `src/`: Source code for data preprocessing, model training, and prediction.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter

## Usage
#### Preprocess the data:
python src/data_preprocessing.py
#### Train the models:
python src/model_training.py --model lstm
python src/model_training.py --model rnn
#### Make predictions:
python src/prediction.py --model lstm
python src/prediction.py --model rnn
## Dataset
The dataset used for this project should be placed in the data/raw/ directory. The data should be a CSV file containing historical energy consumption records.

## Model Training
The models are defined in the models/ directory. The training scripts can be found in src/model_training.py. You can specify which model to train (LSTM or RNN) by passing the --model argument.

## Results
Model evaluation results and plots are saved in the results/ directory. This includes performance metrics and visualizations of the predictions compared to the actual energy consumption.
