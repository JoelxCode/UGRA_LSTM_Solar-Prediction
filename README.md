# 🌞 LSTM Solar Power Prediction

## 📘 Summary / Abstract
This project applies **Long Short-Term Memory (LSTM)** networks to forecast short-term solar photovoltaic (PV) power generation.  
Using weather and irradiance data (BigData.csv, MedData.csv, SmallData.csv), the project builds upon a baseline LSTM model and explores improvements in preprocessing, architecture, and hyperparameter optimization.  

The workflow follows a systematic pipeline:
1. Run and analyze the baseline model  
2. Explore data preprocessing techniques  
3. Adjust the LSTM architecture  
4. Optimize hyperparameters  
5. Evaluate model performance and compare results  

Evaluation metrics include **MAE (Mean Absolute Error)**, **RMSE (Root Mean Squared Error)**, and **R² (Coefficient of Determination)**.

---

## Step 1: Run and Analyze the Baseline Model
- Train the baseline LSTM model with default parameters.  
- Record initial performance (MAE, RMSE, R²).  
- Use these results as the benchmark for later improvements.  

Example:
```python
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

```

## Step 2: Explore Data Preprocessing Techniques
- Handle missing values (removal of outliers)
- Normalize and standarzie features to improve the training stability.
- Experiment with feature engineering.

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
