import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader as data
from keras.models import load_model
import streamlit as stream

stream.title('Predicting Stock Prices')

ticker = stream.text_input("Provide name of Stock")
start_date = '2010-01-01'
end_date = '2021-07-31' 
