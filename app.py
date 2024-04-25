import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from tensorflow.keras.models import load_model # type: ignore
import streamlit as stream
from sklearn.preprocessing import MinMaxScaler

try:
    stream.title('Predicting Stock Prices')

    ticker = stream.text_input("Provide name of Stock Ticker")
    start_date = '2010-01-01'
    end_date = '2021-01-01' 

    frame = yf.download(ticker, start_date, end_date)
    if frame is not None and not frame.empty:
        stream.subheader(f"Stock Data of {ticker} from 2010 - 2021")
        stream.write(frame.describe())

        stream.write(f"<span style='background-color: yellow;'>Closing Price: Final price at which {ticker} stock was traded on a given trading day</span>", unsafe_allow_html=True)
        stream.write("<span style='background-color: yellow;'>Moving Average of x: Takes the average of all data points in a window of size x, adn repeats for all subsets of size x</span>",  unsafe_allow_html=True)

        stream.subheader("Closing Price Over Time")
        clp = plt.figure(figsize = (14, 7))
        plt.plot(frame.Close)
        stream.pyplot(clp)

        stream.subheader("Closing Price Over Time w/ Moving Average of 100")
        stream.write("The blue line represents the closing price over time. The green line represents the closing price over with with a moving average of 100.")
        moving_av_100 = frame.Close.rolling(100).mean()
        clp2 = plt.figure(figsize = (14, 7))
        plt.plot(moving_av_100, 'g')
        plt.plot(frame.Close)
        plt.legend()
        stream.pyplot(clp2)

        stream.subheader("Closing Price Over Time w/ Moving Average of 100 & 200")
        stream.write("The blue line represents the closing price over time. The green line represents the closing price over time with a moving average of 100. The red line represents the closing price over time with a moving average of 200.")
        moving_av_100 = frame.Close.rolling(100).mean()
        moving_av_200 = frame.Close.rolling(200).mean()
        clp3 = plt.figure(figsize = (14, 7))
        plt.plot(moving_av_100, 'g')
        plt.plot(moving_av_200, 'r')
        plt.plot(frame.Close)
        stream.pyplot(clp3)

        #split into training and testing
        data_training = pd.DataFrame(frame['Close'][0:int(len(frame)*0.70)])
        data_testing = pd.DataFrame(frame['Close'][int(len(frame)*0.7): int(len(frame))])

        print(data_training.shape)
        print(data_testing.shape)

        #scale to treat all features equally (values will be scaled down to 0-1)
        scaler = MinMaxScaler(feature_range=(0,1))
        data_training_array = scaler.fit_transform(data_training)

        model = load_model('keras_model.h5')
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.fit_transform(final_df)
        
        x_test = [] #input
        y_test = [] #target output

        #value of next day (y_train) will be dependent on previous 100 days (x_train)
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100: i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        y_predicted = model.predict(x_test)
        scaler = scaler.scale_

        scale_fac = 1/scaler[0]
        y_predicted = y_predicted * scale_fac
        y_test = y_test * scale_fac

        stream.subheader("Predicted Prices vs Original Prices")
        fnal_predictions = plt.figure(figsize=(12,6))
        plt.plot(y_test, 'g', label = 'Original Price')
        plt.plot(y_predicted, 'r', label = 'Predicted Price')
        plt.xlabel('Time (# of Trading Days)')
        plt.ylabel('Price')
        plt.legend()
        stream.pyplot(fnal_predictions)

    else:
        stream.write("Dataframe is empty.")
except Exception as ex:
    stream.write(f"")
    print(ex)
