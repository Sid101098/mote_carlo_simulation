# Pump Sensor Data Analysis with Monte Carlo Simulation and LSTM Prediction

## Overview
This project implements a robust Monte Carlo simulation framework for industrial pump sensor data, followed by LSTM-based predictive modeling. The system:
1. Handles missing data and normalizes sensor readings
2. Performs correlation-preserving Monte Carlo simulations
3. Prepares data for LSTM neural networks to predict future sensor values

## Key Features
- **Data Preprocessing**: Smart handling of missing values and normalization
- **Multivariate Simulation**: Maintains cross-sensor correlations during simulation
- **Conditional Probability**: Uses Schur complement for accurate conditional distributions
- **LSTM Integration**: Ready-to-use data preparation for sequence prediction

## Requirements.
```bash
Python 3.8+
numpy
pandas
scikit-learn
scipy
matplotlib
