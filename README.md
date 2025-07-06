# Bitcoin Price Prediction with LSTM

A Long Short-Term Memory (LSTM) neural network implementation for predicting Bitcoin prices using TensorFlow/Keras and historical price data.

## Project Overview

- **Prediction Accuracy**: R² score of ~0.98 on test data
- **Model Size**: ~118K parameters
- **Training Time**: ~10 minutes (10 epochs)
- **Dataset**: 10 years of Bitcoin daily prices (3,653 records)

## Complete Pipeline Flow

```
                                  ┌─────────────────┐
                                  │  Yahoo Finance  │
                                  │   BTC-USD Data  │
                                  └────────┬────────┘
                                           │
                                           ▼
                                  ┌─────────────────┐     
                                  │  10 Years Data  │     
                                  │  OHLCV Format   │     
                                  └────────┬────────┘     
                                           │
                                           ▼
                                  ┌─────────────────────────────┐
                                  │   Exploratory Analysis      │
                                  │                             │
                                  │   3,653 Records(No missing) │
                                  │   Price: $210 to $111673    │
                                  │                             │
                                  └──────────────┬──────────────┘
                                                 │
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │    Data Preprocessing       │
                                  │  • Select Close Price       │
                                  │  • MinMax Scaling [0,1]     │
                                  └──────────────┬──────────────┘
                                                 │
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │    Sequence Creation        │
                                  │  • 100-day windows → 1 day  │
                                  │  • 3,553 sequences total    │
                                  └──────────────┬──────────────┘
                                                 │
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │      Train-Test Split       │
                                  │  ┌──────────┬─────────────┐ │
                                  │  │ Training │    Test     │ │
                                  │  │   90%    │     10%     │ │
                                  │  │  3,197   │     356     │ │
                                  │  └──────────┴─────────────┘ │
                                  └──────────────┬──────────────┘
                                                 │
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │      Model Building         │
                                  │  ┌─────────────────────┐    │
                                  │  │ Input (100×1)       │    │
                                  │  ├─────────────────────┤    │
                                  │  │ LSTM (128 units)    │    │
                                  │  ├─────────────────────┤    │
                                  │  │ LSTM (64 units)     │    │
                                  │  ├─────────────────────┤    │
                                  │  │ Dense (25)          │    │
                                  │  ├─────────────────────┤    │
                                  │  │ Dense (1)           │    │
                                  │  └─────────────────────┘    │
                                  └──────────────┬──────────────┘
                                                 │
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │      Model Training         │
                                  │  • Optimizer: Adam          │
                                  │  • Loss: MSE                │
                                  │  • Epochs: 10               │
                                  │  • Batch Size: 5            │
                                  └──────────────┬──────────────┘
                                                 │
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │      Model Evaluation       │
                                  │  ┌───────────────────────┐  │
                                  │  │ • Actual vs Predicted │  │
                                  │  │ • RMSE, R², MAE       │  │
                                  │  │ • Future Predictions  │  │
                                  │  └───────────────────────┘  │
                                  └─────────────────────────────┘
                                                                                                                                    
```

## Detailed Process Flow

### 1. **Data Collection**
- Fetch 10 years of Bitcoin price data from Yahoo Finance
- Download includes Open, High, Low, Close, and Volume
- Data spans from 2015 to 2025 with 3,653 daily records

### 2. **Exploratory Data Analysis**
- **Price Range**: $210 to $111,000 
- **Volatility**: High standard deviation indicating market turbulence
- **Trend Analysis**: 100-day and 365-day moving averages calculated
- **Data Quality**: No missing values found

### 3. **Data Preprocessing**
- **Feature Selection**: Focus on closing price for univariate prediction
- **Normalization**: MinMax scaling transforms prices to [0,1] range
- **Purpose**: Improves neural network convergence and prevents gradient issues

### 4. **Sequence Preparation**
- **Sliding Window**: Use 100 consecutive days to predict next day
- **Total Sequences**: 3,553 input-output pairs created
- **Overlap**: Maximizes available training data through overlapping windows

### 5. **Model Architecture**
The LSTM architecture consists of:
- **First LSTM Layer (128 units)**:
  - Processes sequential data with memory cells
  - Returns sequences for next LSTM layer
  - Captures initial temporal patterns
- **Second LSTM Layer (64 units)**:
  - Refines temporal features
  - Returns single output vector
- **Dense Layers**:
  - Hidden layer (25 neurons) for non-linear transformation
  - Output layer (1 neuron) for price prediction

### 6. **Model Training**
- **Optimizer**: Adam with adaptive learning rates
- **Loss Function**: Mean Squared Error for regression
- **Training Configuration**: 10 epochs, batch size of 5

### 7. **Model Evaluation**
- **Test Performance**: Model captures general price trends
- **Metrics Used**:
  - RMSE: Root Mean Squared Error in dollar terms
  - R² Score: Proportion of variance explained (~0.98)
  - MAE: Mean Absolute Error for average deviation
- **Future Predictions**: Recursive 10-day forecast with increasing uncertainty

## Key Results

| Metric   |     Description        | Performance |
|----------|------------------------|-------------|
| R² Score | Variance explained     | ~0.98       |
| RMSE     | Root Mean Square Error | 2190.06     |
| MAE      | Mean Absolute Error    | 1611.47     |

