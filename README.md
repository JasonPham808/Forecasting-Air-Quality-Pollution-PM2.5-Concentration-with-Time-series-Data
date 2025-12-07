
# Forecasting PM2.5 Air Quality Forecasting with LSTM Project

This repository contains a multivariate time-series forecasting project that predicts hourly PM2.5 concentrations using Long Short-Term Memory (LSTM) neural networks. The model is designed to support air-quality monitoring, public health planning, and urban environmental management by providing short-term PM2.5 forecasts based on historical pollution and meteorological data.[1]

***

## 1. Project Overview

### 1.1 Problem Statement

Fine particulate matter (PM2.5) is a key air pollutant linked to respiratory illness, premature mortality, and reduced visibility in urban environments. The objective of this project is to forecast PM2.5 levels for the next 24 hours using the previous 72 hours of data, enabling proactive responses such as health advisories and pollution control measures.[1]

### 1.2 Data Summary

The dataset consists of hourly observations from March 2013 to February 2017. Each record includes PM2.5, several gaseous pollutants, and meteorological variables. The main columns are:[1]

- `datetime` – timestamp (later converted to a DateTime index)  
- `PM2.5` – fine particulate matter concentration (target variable)  
- `SO2` – sulfur dioxide  
- `NO2` – nitrogen dioxide  
- `CO` – carbon monoxide  
- `O3` – ozone  
- `TEMP` – temperature  
- `PRES` – pressure  
- `DEWP` – dew point  
- `WSPM` – wind speed  

A subset of these features is used for modeling after correlation analysis and feature selection.[1]

***

## 2. Repository Structure

A recommended structure for this project is:

```text
.
├── data/
│   └── Task_3_Dataset_Air_Quality.csv
├── notebooks/
│   └── air_quality_lstm.ipynb
├── src/
│   ├── preprocessing.py
│   ├── modeling.py
│   └── evaluation.py
├── README.md
└── requirements.txt
```

- `data/` holds the raw dataset.  
- `notebooks/` contains the exploratory and development notebook.  
- `src/` groups reusable preprocessing, modeling, and evaluation code.  
- `requirements.txt` lists the Python dependencies.[1]

***

## 3. Installation & Requirements

### 3.1 Dependencies

The project uses the following key libraries:[1]

- `numpy`, `pandas` – numerical & tabular data handling  
- `matplotlib`, `seaborn` – visualisation  
- `scikit-learn` – scaling and evaluation metrics  
- `tensorflow` / `keras` – deep learning (LSTM implementation)

Example `requirements.txt`:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
```

### 3.2 Setup

```bash
git clone <your-repo-url>.git
cd <your-repo-folder>

pip install -r requirements.txt
```

Place the dataset file in `data/Task_3_Dataset_Air_Quality.csv`.[1]

***

## 4. End-to-End Pipeline

This section explains each major step in the project pipeline, from raw data to model evaluation.

### 4.1 Data Loading & Initial Exploration

1. **Load the CSV file**

```python
import pandas as pd

dataset = pd.read_csv("data/Task_3_Dataset_Air_Quality.csv")
```

2. **Inspect structure and quality**

- Use `dataset.info()` to check dtypes and missing values.  
- Use `dataset.describe()` to examine the distribution and range of each feature.[1]

3. **Convert and set datetime index**

```python
dataset['datetime'] = pd.to_datetime(
    dataset['datetime'],
    format='%d/%m/%Y %H:%M',
    errors='coerce'
)
dataset = dataset.set_index('datetime')
```

Using a DateTime index enables time-based slicing, chronological train/test splitting, and time-series visualisation.[1]

4. **Correlation and basic plots**

- Compute a correlation matrix to see which variables are most related to PM2.5.  
- Plot a heatmap to visually inspect correlations.  
- Create individual line plots for PM2.5 and other pollutants to understand temporal patterns.[1]

***

### 4.2 Feature Selection & Missing Value Handling

#### 4.2.1 Feature Selection

Correlation analysis shows that `SO2`, `NO2`, and `CO` have noticeably stronger correlations with PM2.5 than the meteorological variables and ozone. The following features are retained for modeling:[1]

- `PM2.5`  
- `SO2`  
- `NO2`  
- `CO`  

The remaining features (`O3`, `TEMP`, `PRES`, `DEWP`, `WSPM`) are dropped to reduce dimensionality and focus on the most relevant predictors.[1]

```python
columns_to_drop = ['O3', 'TEMP', 'PRES', 'DEWP', 'WSPM']
dataset = dataset.drop(columns=columns_to_drop)
```

#### 4.2.2 Missing Values

The main missing data issue concerns `NO2`, with around 692 missing entries. Rather than dropping rows (which would break time continuity and shrink the dataset), the mean value of `NO2` is used for imputation. This preserves sequence integrity and avoids bias from row deletion.[1]

```python
dataset['NO2'] = dataset['NO2'].fillna(dataset['NO2'].mean())
```

A histogram of `NO2` values is used to confirm that the distribution is reasonable and that mean imputation is acceptable.[1]

***

### 4.3 Time-based Train/Test Split

Instead of random splitting, the dataset is divided chronologically:

- **Training period**: all data before 2016  
- **Test period**: all data from 2016 onwards[1]

This simulates real-world forecasting, where models are trained on past data and evaluated on future observations.

```python
def train_test_split(dataset, split_year):
    train = dataset.loc[dataset.index.year < split_year]
    test = dataset.loc[dataset.index.year >= split_year]
    return train, test

training_set, test_set = train_test_split(dataset, 2016)
```

The resulting shapes are:[1]

- Training set: 24,864 rows, 4 columns  
- Test set: 10,200 rows, 4 columns  

***

### 4.4 Scaling & Sequence Construction

#### 4.4.1 Feature Scaling

LSTMs are sensitive to feature scale, so all features are normalised to the  range using `MinMaxScaler`.[2][1]

```python
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
```

Scaling ensures that large-magnitude variables (e.g., CO) do not dominate training dynamics.[1]

#### 4.4.2 Sliding Window Sequences

The forecasting task uses a sliding window strategy:

- **Input window (`n_steps`)**: previous 72 hours  
- **Forecast horizon (`forecasting_horizon`)**: next 24 hours  
- **Features per timestep**: 4 (PM2.5, SO2, NO2, CO)[1]

```python
import numpy as np

def split_sequence(sequence, n_steps, forecasting_horizon, y_index):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - forecasting_horizon:
            break
        seq_x = sequence[i:end_ix]
        seq_y = sequence[end_ix:end_ix + forecasting_horizon, y_index]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

n_steps = 72
forecasting_horizon = 24
y_index = 0  # assuming PM2.5 is the first column after scaling

X_train, y_train = split_sequence(training_set_scaled, n_steps, forecasting_horizon, y_index)
```

The resulting training shapes are approximately:[1]

- `X_train`: `(24769, 72, 4)`  
- `y_train`: `(24769, 24)`  

This structure is appropriate for feeding into an LSTM: sequences of 72 timesteps with 4 features.[1]

***

### 4.5 LSTM Model Architecture

The best model (referred to as *Model 11* in the report) is a deep stacked LSTM network tailored for multivariate sequence-to-sequence forecasting.[1]

**Architecture highlights:**

- Input shape: `(72, 4)`  
- LSTM Layer 1: 256 units, `tanh` activation, returns full sequences, L2 regularisation, dropout 0.2  
- LSTM Layer 2: 128 units, `tanh` activation, returns full sequences, L2 regularisation, dropout 0.2  
- LSTM Layer 3: 64 units, `tanh` activation, returns final output, L2 regularisation, dropout 0.2  
- Dense output layer: 24 units (one for each forecast hour), linear activation  
- Optimizer: Adam with a low learning rate (e.g., 0.0001)  
- Loss: Mean Squared Error (MSE)[1]

This depth allows the model to extract both short- and long-term temporal patterns across the pollutants. Regularisation and dropout control overfitting.[1]

***

### 4.6 Model Training

The model is trained using mini-batch gradient descent:[1]

- Epochs: up to 100  
- Batch size: 64  

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

model = Sequential()
model.add(Input(shape=(n_steps, X_train.shape[2])))
model.add(LSTM(256, activation='tanh', return_sequences=True, kernel_regularizer=l2(0.001)))
model.add(Dropout(0.2))
model.add(LSTM(128, activation='tanh', return_sequences=True, kernel_regularizer=l2(0.001)))
model.add(Dropout(0.2))
model.add(LSTM(64, activation='tanh', return_sequences=False, kernel_regularizer=l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(forecasting_horizon))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
history = model.fit(X_train, y_train, epochs=100, batch_size=64)
```

During training, the loss steadily decreases and stabilises at a low value, reflecting the model’s ability to fit the temporal patterns without explosive overfitting.[1]

***

### 4.7 Testing, Inverse Scaling & Evaluation

#### 4.7.1 Test Data Preparation

The test set undergoes the same scaling and sequence generation as the training set, using the scaler fitted on the training data to prevent data leakage.[1]

```python
test_scaled = sc.transform(test_set)
X_test, y_test = split_sequence(test_scaled, n_steps, forecasting_horizon, y_index)
```

#### 4.7.2 Target Rescaling

A separate scaler is fitted to the training targets to accurately invert predictions back to the original PM2.5 scale.[1]

```python
from sklearn.preprocessing import MinMaxScaler

target_scaler = MinMaxScaler()
y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1))

pred_scaled = model.predict(X_test)
pred_pm25 = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(pred_scaled.shape)

# Rescale ground truth for fair comparison
y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
```

#### 4.7.3 Mean Absolute Error (MAE)

MAE is computed for each forecasting step and averaged across the 24-hour horizon.[1]

```python
from sklearn.metrics import mean_absolute_error
import numpy as np

mae_per_horizon = [
    mean_absolute_error(y_test_rescaled[:, i], pred_pm25[:, i])
    for i in range(forecasting_horizon)
]
average_mae = np.mean(mae_per_horizon)
print(f"Average MAE across all horizons: {average_mae:.2f}")
```

The model achieves an average MAE of about **0.04** over all 24 forecast steps, demonstrating stable performance across short and longer horizons.[1]

***

### 4.8 Visualisation of Forecasts

#### 4.8.1 Single-sample view

For a chosen sample, the project compares:

- The 72-hour input history  
- The true 24-hour PM2.5 trajectory  
- The predicted 24-hour PM2.5 trajectory[1]

This illustrates that the model captures general level and trend but can underestimate the sharpness of sudden drops to near-zero values.

#### 4.8.2 Horizon-wise comparison

A simple plotting function displays true vs predicted PM2.5 over one full 24-hour forecast window:[1]

```python
import matplotlib.pyplot as plt

def plot_predictions(true_values, predicted_values):
    plt.plot(true_values, color="gray", label="True PM2.5")
    plt.plot(predicted_values, color="red", label="Predicted PM2.5")
    plt.title("PM2.5 – True vs Predicted")
    plt.xlabel("Hours Ahead")
    plt.ylabel("PM2.5 Concentration")
    plt.legend()
    plt.show()
```

Plots show that predictions track the overall magnitude but may smooth out extreme variations.[1]

***

## 5. Key Insights & Learnings

- **Correlated pollutants matter**: Focusing on `SO2`, `NO2`, and `CO` alongside PM2.5 improves signal quality relative to including weakly related variables.[1]
- **Temporal design is crucial**: A 72-hour input window and 24-hour horizon provide a good balance between context and forecast usefulness.[1]
- **Depth and regularisation pay off**: A 3-layer LSTM with dropout and L2 regularisation (Model 11) delivers the most stable and accurate results among tested configurations.[1]
- **Performance is practically useful**: An average MAE around 0.04 suggests the model is suitable for operational early-warning systems, especially when combined with domain-specific thresholds.[1]
- **Limitations remain**: The model can struggle with abrupt regime shifts, tending to over-predict during rapid air-quality improvements, which motivates future work on attention mechanisms or hybrid models.[1]

***

## 6. Real-world Applications

This forecasting pipeline can be integrated into:[1]

- **Public health dashboards** – to visualise upcoming pollution episodes.  
- **Alert systems** – to trigger warnings for haze and high-pollution conditions.  
- **Policy tools** – to support short-term traffic and industrial restrictions.  

Enhancements could include spatial data from multiple stations, attention-based models, or ensemble architectures to further increase robustness.[1]

***

## 7. How to Run This Project

1. Clone the repository and install dependencies.  
2. Place the air quality dataset into `data/Dataset_Air_Quality.csv`.  
3. Open and run `notebooks/air_quality_lstm.ipynb` step by step:
   - Data loading and EDA  
   - Preprocessing and scaling  
   - Sequence generation  
   - LSTM model definition and training  
   - Evaluation and visualisation  
