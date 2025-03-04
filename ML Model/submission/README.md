
# Stock Price Prediction Using Machine Learning Models

## Overview
This project predicts stock price movements by training machine learning models using technical indicators. The models include:
- LightGBM
- XGBoost
- CatBoost

The target is to predict percentage returns for the next 15 minutes of stock price data.

---

## Features
- **Data Preprocessing**:
  - Technical indicators such as SMA, EMA, MACD, RSI, and more are calculated.
  - Percentage returns are calculated as the target variable.
- **Model Training**:
  - Models are trained on a rolling window of stock price data.
  - Supports cross-validation with configurable folds.
- **GPU Support**:
  - Utilizes GPU acceleration for training models like LightGBM, XGBoost, and CatBoost.
- **Prediction**:
  - Predictions are averaged across multiple models for better accuracy.

---

## Installation
### Prerequisites
- Python 3.8 or later

### Install Dependencies
Make sure you have all the required libraries installed. Use the following command to install dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. **Set up the environment**
Modify the following variables in the script as needed:
- `stock`: Set to `"BTC"` or `"ETH"` for the desired stock.
- `interval`: Set the data interval, e.g., `"1min"`.
- `global_input_path_train`: Path to the training dataset.
- `global_input_path_test`: Path to the test dataset.

> **Note:**  
> If you are running the code on any other random dataset, make sure to:
> - Change the paths for `global_input_path_train` and `global_input_path_test` to point to your dataset:
>   ```python
>   global_input_path_train = "... set the input path of the file that has train data"
>   global_input_path_test = "... set the input path of the file that has test data"
>   ```
>   Update this in the `# In[2]: If in training mode, load the training data and test data` cell of the code.
> - Adjust the `num_valid_dates` variable to suit the size and nature of your dataset.
> - Change the `test_data_points=10000` in the `# In[8]: loading the models and predicting on the test data` cell to match the size of the test dataset.

### 2. **Run the script**
```bash
python submission.py
```

### 3. **Outputs**
- Trained models are saved in the `models/` directory.
- Predictions are saved in a CSV file:
  - For BTC: `BTC_test_predicted.csv`
  - For ETH: `ETH_test_predicted.csv`

---

## Model Details

### LightGBM
- Objective: `l2`
- GPU support enabled.

### XGBoost
- Objective: `reg:squarederror`
- GPU accelerated training with the `"hist"` method.

### CatBoost
- Objective: `RMSE`
- GPU support enabled.

---

## Metrics
- **R² Score**: Measures the goodness of fit for model predictions on test data.

---

## File Structure
- **`submission.py`**: Main script containing data preprocessing, model training, and prediction logic.
- **`models/`**: Directory to store trained models.
- **Input Data Requirements**:
  - **For BTC**:
    - Input file must be in the folder named `btc-1-min-data`.
    - It should be a **single file** containing both training and test datasets.
    - The last 1,00,000 rows will be used for testing.
    - Ensure the test dataset size (`test_data_points`) in the code is set to `100000` in the `# In[8]: loading the models and predicting on the test data` section.
  - **For ETH**:
    - Input files must be inside the folder named `eth-1-min-data`.
    - Files should follow the specific format, e.g., `2020_21_1min_eth.csv`, with one file per year.
    - The last file, i.e., `2023_24_1min_eth.csv`, will be used as the test dataset.
    - Only the first 10,000 rows from this file will be used for testing.
    - Ensure the test dataset size (`test_data_points`) in the code is set to `10000` in the `# In[8]: loading the models and predicting on the test data` section.

---

## Troubleshooting
1. **Special Characters in Feature Names**:
   Ensure feature names do not contain special characters such as `.` or `,`.
2. **NaN Values in Data**:
   Missing or invalid values in datetime columns are logged and handled.

---
