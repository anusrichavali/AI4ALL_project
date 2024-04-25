import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader as data
import keras_models
import streamlit as stream
from pandas_datareader._utils import RemoteDataError

stream.title('Predicting Stock Prices')

ticker = stream.text_input("Provide name of Stock")
start_date = '2010-01-01'
end_date = '2021-12-31' 

try:
    frame = data.DataReader(ticker, 'yahoo', start_date, end_date)
    print("Frame:", frame)
    if frame is not None:
        if not frame.empty:
            stream.write(frame.describe())
        else:
            raise RemoteDataError
except Exception as ex:
    stream.write(f"Error: {ex}")