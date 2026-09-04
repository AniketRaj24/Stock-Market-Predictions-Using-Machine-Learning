# 📈 Stock Market Price Prediction using LSTM

A deep learning web app that predicts stock closing prices from historical data using a 4-layer stacked LSTM network, with an interactive **Streamlit** dashboard for visualization.

Live demo: fetch any stock ticker (e.g. `GOOG`, `AAPL`, `TSLA`) and see moving averages plus a model-predicted price trend against the real one — all in the browser.

---

## 🖼️ Results

### Original vs. Predicted Price (GOOG, test set)
![Original vs Predicted Price](assets/original_vs_predicted.png)

The model captures the overall *direction* and *shape* of price movement, but consistently trails the true price and slightly underestimates magnitude — a common and expected pattern for LSTM models trained on raw closing price alone. See [Limitations](#-limitations--honest-notes) below for why, and what would fix it.

### Price with 100 & 200-Day Moving Averages
![MA100 vs MA200 vs Close](assets/ma100_ma200_vs_close.png)

---

## 🧠 Model Architecture

A stacked LSTM regressor built in Keras, trained to predict the next day's closing price from the previous 100 days:

```
Input: (100 timesteps, 1 feature — Close price)

LSTM(50,  activation='relu', return_sequences=True) → Dropout(0.2)
LSTM(60,  activation='relu', return_sequences=True) → Dropout(0.3)
LSTM(80,  activation='relu', return_sequences=True) → Dropout(0.4)
LSTM(120, activation='relu')                        → Dropout(0.5)
Dense(1)  # predicted next-day closing price
```

| | |
|---|---|
| **Total parameters** | 536,285 |
| **Trainable parameters** | 178,761 |
| **Loss function** | Mean Squared Error (MSE) |
| **Optimizer** | Adam |
| **Epochs** | 50 |
| **Batch size** | 32 |
| **Lookback window** | 100 days |
| **Scaling** | Min-Max Normalization (0–1) |

The increasing Dropout rate at each layer (0.2 → 0.5) is a deliberate regularization strategy — deeper layers carry more parameters and are more prone to overfitting on a relatively small, noisy financial time series.

---

## 📊 Dataset

- **Source:** [Yahoo Finance](https://finance.yahoo.com) via the `yfinance` API
- **Ticker used for training/evaluation:** `GOOG` (Alphabet Inc.)
- **Date range:** Jan 1, 2012 – Dec 21, 2022 (2,761 trading days)
- **Feature used:** Daily closing price only (single-variate)
- **Split:** 80% train (2,208 days) / 20% test (553 days), chronological — no shuffling, since shuffling time series data leaks future information into training

The trained `.keras` model is bundled in the repo, but the Streamlit app (`app.py`) re-fetches live data for whatever ticker the user enters, so it works beyond just GOOG (with the caveat that the model was only *trained* on GOOG — see below).

---

## 📉 Evaluation Metrics

Measured on the held-out test set (unseen during training):

| Metric | Value |
|---|---|
| **RMSE** | 28.75 |
| **MAE** | 26.07 |

For context, GOOG traded roughly between $85–$150 over the test window, so this represents meaningful average error — the model is directionally useful but not precise enough for real trading decisions. This is reported honestly rather than cherry-picked, and is discussed further below.

---

## ⚠️ Limitations & Honest Notes

This project is a learning/portfolio project, not a production trading system. Worth being upfront about:

- **Single-feature input.** The model only sees closing price — no volume, no technical indicators (RSI, MACD, Bollinger Bands), no macro or sentiment data. Adding these is a natural next step.
- **Lag effect.** Like most next-step LSTM predictors trained this way, the model's predictions lag the actual price — it tends to echo yesterday's trend rather than truly forecast tomorrow's. This is visible in the chart above.
- **Single-stock training.** The model was trained only on GOOG; predictions for other tickers via the app will be less reliable since the model hasn't learned their specific volatility patterns.
- **No walk-forward/backtesting.** Evaluation is a single train/test split, not rolling-window backtesting, so the reported RMSE/MAE is one snapshot, not a robust estimate across market regimes.
- **Stock prices are notoriously hard to predict** from price history alone — markets are close to efficient, and this project is best understood as an exercise in applying LSTMs to time series, not a claim that it beats the market.

### Ideas for improvement
- Multivariate input (Open/High/Low/Volume + technical indicators)
- Walk-forward validation instead of a single split
- Compare against simpler baselines (ARIMA, linear regression) to quantify what the LSTM actually adds
- Predict returns/direction instead of raw price, which is often more tractable than exact price regression

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/AniketRaj24/Stock-Market-Predictions-Using-Machine-Learning.git
cd Stock-Market-Predictions-Using-Machine-Learning
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```
Then open **http://localhost:8501** in your browser.

---

## 🗂️ Project Structure

```
Stock-Market-Predictions-Using-Machine-Learning/
│
├── app.py                                              # Streamlit web app (inference + visualization)
├── Stock_Market_Prediction_Model_Creation_Updated.ipynb # Training notebook (data prep → model → evaluation)
├── Stock_Predictions_Model.keras                        # Trained LSTM weights
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

`Python` · `TensorFlow / Keras` · `Streamlit` · `yfinance` · `NumPy` · `pandas` · `scikit-learn` · `Matplotlib`

---

## 📄 License

Add a license (MIT is a common default for portfolio projects) so others know how they can use this code.
