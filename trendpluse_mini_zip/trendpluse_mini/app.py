from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime
import time
import random
import yfinance as yf
from flask_socketio import SocketIO

# Initialize Flask app
app = Flask(__name__)
socketio = SocketIO(app)

# Load the saved model
model_path = r"C:\Users\Bhanu prakash Reddy\Downloads\trendpluse_mini_zip\trendpluse_mini\Latest_stock_price_model.keras"
model = load_model(model_path)

# Define the route for the home page
@app.route('/')
def index():
    current_datetime = datetime.now()
    return render_template('index.html', datetime=current_datetime)

# Define the route for AI predictions
@app.route('/ai-predictions', methods=['GET', 'POST'])
def ai_predictions_result():
    if request.method == 'POST':
        # Retrieve form data
        price = request.form['price']
        selected_gadgets = request.form.getlist('gadgets')
        region = request.form['region']
        product = request.form['product']
        
        # File path to the dataset
        file_path = r"C:\Users\vmgow\OneDrive\Desktop\trendpluse_mini\templates\trendpluse_dataset(csv).csv"
        
        # Load and preprocess the data
        data = pd.read_csv(file_path)
        data['pct_change'] = (data['Close'] - data['Open']) / data['Open'] * 100
        data['7_day_avg'] = data['Close'].rolling(window=7).mean()
        data['30_day_avg'] = data['Close'].rolling(window=30).mean()
        data['volatility'] = data['Close'].rolling(window=30).std()
        data['target'] = np.where(data['Close'].shift(-180) > data['Close'], 1, 0)
        data = data.dropna()

        # Define features and target
        features = ['Open', 'High', 'Low', 'Close', 'Adjusted', 'Volume', 'Dividend amount', 'pct_change', '7_day_avg', '30_day_avg', 'volatility']
        X = data[features]
        y = data['target']

        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions on the latest data
        latest_data = X.tail(1)
        latest_data_scaled = scaler.transform(latest_data)
        prediction = model.predict(latest_data_scaled)

        # Prepare prediction message and recommendation
        if prediction[0] == 1:
            prediction_message = "The stock is predicted to increase in the next 6 months."
            recommendation_message = "It's a great time to consider launching new products to take advantage of the favorable market conditions."
            percentage_decrease = None
        else:
            prediction_message = "The stock is predicted to decrease in the next 6 months."
            old_value = data['Close'].iloc[-2]
            new_value = data['Close'].iloc[-1]
            percentage_decrease = ((old_value - new_value) / old_value) * 100
            recommendation_message = (
                f"With an expected decrease of {percentage_decrease:.2f}%, it might not be the best time to launch new products. "
                "Consider focusing on discount sales or promotions to maintain customer interest and sales volume."
            )

        # Simulate percentage changes for the next 6 months
        predicted_changes = [np.random.uniform(-5, 5) for _ in range(6)]
        months = ['Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6']

        # Plot the predicted changes
        plt.figure(figsize=(10, 5))
        plt.plot(months, predicted_changes, marker='o', linestyle='-', color='b', label='Predicted Change')
        plt.title("Predicted Stock Price Changes for Next 6 Months")
        plt.xlabel('Month')
        plt.ylabel('Percentage Change (%)')
        plt.grid(True)
        plt.legend()

        # Save plot to a string buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        # Render the template with predictions and plot
        return render_template(
            'ai-predictions-result.html',
            region=region,
            product=product,
            price=price,
            gadgets=selected_gadgets,
            prediction_message=prediction_message,
            recommendation_message=recommendation_message,
            percentage_decrease=percentage_decrease,
            plot_data=plot_data
        )

    return render_template('ai-predictions.html', gadgets=["Gadget 1", "Gadget 2", "Gadget 3", "Gadget 4", "Gadget 5", "Gadget 6", "Gadget 7", "Gadget 8"])
# Define the route for Market Analysis
@app.route('/market-analysis', methods=['GET', 'POST'])
def market_analysis():
    if request.method == 'POST':
        company_code = request.form['company_code']
        
        # Retrieve historical stock data for the given company code
        end = datetime.now()
        start = datetime(end.year - 20, end.month, end.day)
        stock_data = yf.download(company_code, start, end)

        # Preprocess the data
        Adj_close_price = stock_data[['Adj Close']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(Adj_close_price)
        
        x_data = []
        y_data = []
        for i in range(100, len(scaled_data)):
            x_data.append(scaled_data[i-100:i])
            y_data.append(scaled_data[i])

        x_data, y_data = np.array(x_data), np.array(y_data)

        # Make predictions using the loaded model
        predictions = model.predict(x_data)
        inv_predictions = scaler.inverse_transform(predictions)

        # Prepare actual vs predicted data for plotting
        inv_y_data = scaler.inverse_transform(y_data)
        
        plot_data = pd.DataFrame({
            'Original Test Data': inv_y_data.flatten(),
            'Predictions': inv_predictions.flatten()
        }, index=stock_data.index[100:])
        
        # Plotting example
        plt.figure(figsize=(15, 5))
        plt.plot(plot_data.index, plot_data['Original Test Data'], color='blue', label='Original Data')
        plt.plot(plot_data.index, plot_data['Predictions'], color='red', label='Predicted Data')
        plt.title(f"Original vs Predicted Stock Prices for {company_code}")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data_encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        return render_template(
            'market-analysis-result.html',
            company_code=company_code,
            prediction_message="Here are the predicted stock prices based on the model.",
            plot_data=plot_data_encoded
        )

    return render_template('market-analysis.html')

# trading bot
# Simulation parameters
STOCK_SYMBOL = "AAPL"
BUY_THRESHOLD = 170.00
SELL_THRESHOLD = 175.00
TRADE_QUANTITY = 1
BASE_PRICE = 175.00



# To store stock prices for plotting
stock_prices = []
timestamps = []

def simulate_stock_price():
    global BASE_PRICE
    price_change = random.uniform(-1, 1)
    BASE_PRICE += price_change
    BASE_PRICE = max(150, min(BASE_PRICE, 200))
    return BASE_PRICE

def place_order(action, symbol, quantity):
    print(f"Order placed: {action.upper()} {quantity} shares of {symbol}")

def plot_graph():
    plt.clf()
    plt.plot(timestamps, stock_prices, label=f'{STOCK_SYMBOL} Stock Price', color='blue', linewidth=2, marker='o', markersize=4)
    for i in range(1, len(stock_prices)):
        percent_change = ((stock_prices[i] - stock_prices[i - 1]) / stock_prices[i - 1]) * 100
        x_pos = timestamps[i]
        y_pos = stock_prices[i]
        change_text = f"{percent_change:+.2f}%"
        plt.text(x_pos, y_pos, change_text, fontsize=8, color='darkgreen' if percent_change >= 0 else 'red', ha='center', va='bottom')
    plt.title(f'{STOCK_SYMBOL} Stock Price Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Stock Price ($)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)

def monitor_and_trade():
    print(f"Monitoring {STOCK_SYMBOL} for trading...")
    while True:
        try:
            current_price = simulate_stock_price()
            current_time = datetime.now().strftime('%H:%M:%S')
            stock_prices.append(current_price)
            timestamps.append(current_time)
            if current_price < BUY_THRESHOLD:
                place_order("buy", STOCK_SYMBOL, TRADE_QUANTITY)
            elif current_price > SELL_THRESHOLD:
                place_order("sell", STOCK_SYMBOL, TRADE_QUANTITY)
            plot_graph()
            time.sleep(5)
        except Exception as e:
            print(f"Error in monitoring: {e}")
            time.sleep(5)

@app.route('/trading-bot')
def trading_bot():
    return render_template('trading-bot.html')

@app.route('/trading-bot-result')
def trading_bot_result():
    plot_graph()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return render_template('trading-bot-result.html', stock_symbol=STOCK_SYMBOL, plot_data=plot_data)

@socketio.on('start_simulation')
def start_simulation():
    while True:
        current_price = simulate_stock_price()
        current_time = datetime.now().strftime('%H:%M:%S')
        action = None
        if current_price < BUY_THRESHOLD:
            place_order("buy", STOCK_SYMBOL, TRADE_QUANTITY)
            action = "Buy"
        elif current_price > SELL_THRESHOLD:
            place_order("sell", STOCK_SYMBOL, TRADE_QUANTITY)
            action = "Sell"
        socketio.emit('stock_update', {'time': current_time, 'price': current_price, 'action': action})
        time.sleep(1)

if __name__ == '__main__':
    plt.ion()
    socketio.run(app, debug=True)

