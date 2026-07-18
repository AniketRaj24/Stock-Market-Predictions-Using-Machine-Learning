#  Stock Market Prediction Using LSTM (Deep Learning)

This project is a **Stock Price Prediction Web Application** built using a **Long Short-Term Memory (LSTM)** neural network.  
It predicts future stock prices based on historical data and visualizes trends using moving averages.

The application is deployed locally using **Streamlit** for interactive visualization.

---

##  Features

- Fetches real-time historical stock data using **Yahoo Finance (yfinance)**
- Uses **LSTM deep learning model** trained on past prices
- Moving Averages visualization:
  - MA50
  - MA100
  - MA200
- Predicts future stock prices
- Displays:
  - Original vs Predicted price graph
  - RMSE error metric

---

##  Model Details

- Architecture: LSTM Neural Network (Keras / TensorFlow)
- Input Features:
  - Closing Price
- Window Size: 100 days
- Scaling: Min-Max Normalization (0–1)
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam

---

##  Project Structure

Stock_Market_Prediction_ML/
│
├── app.py # Streamlit web app
├── Stock_Predictions_Model.keras # Trained LSTM model
├── Stock_Market_Prediction_Model_Creation_Updated.ipynb # Training notebook
├── requirements.txt
└── README.md

---

##  Installation & Setup

### 1️ Clone the Repository

```bash
git clone https://github.com/vibhu1304/Stock-Market-Prediction-using-LSTM-Networks.git
cd Stock_Market_Prediction_ML
2️⃣ Create Virtual Environment (Optional but Recommended)
python -m venv venv
venv\Scripts\activate  
3️⃣ Install Dependencies
pip install -r requirements.txt
▶️ Run the Application
streamlit run app.py
Then open in browser:
http://localhost:8501