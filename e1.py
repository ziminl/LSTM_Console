# Nowa konsola do uruchamiania modeli LSTM
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta, date
from pathlib import Path
import appdirs as ad
CACHE_DIR = ".cache"
# Force appdirs to say that the cache dir is .cache
ad.user_cache_dir = lambda *args: CACHE_DIR
# Create the cache dir if it doesn't exist
Path(CACHE_DIR).mkdir(exist_ok=True)
import yfinance as yf

today = date.today()
comm_dict2 = {'EURUSD=X':'USD_EUR','CNY=X':'USD/CNY','CL=F':'Crude_Oil','GC=F':'Gold','^IXIC':'NASDAQ',
             '^GSPC':'SP_500','^TNX':'10_YB','HG=F':'Copper','GBPUSD=X':'USD_GBP','JPY=X':'USD_JPY',
              'EURPLN=X':'EUR/PLN','PLN=X':'PLN/USD', 'AED=X':'USD/AED','^FVX':'5_YB','RUB=X':'USD/RUB',
              'PL=F':'Platinum','SI=F':'Silver','NG=F':'Natural Gas',
              'ZR=F':'Rice Futures','ZS=F':'Soy Futures','KE=F':'KC HRW Wheat Futures'}

st.title('LSTM Model - main console')

@st.cache_data
def model_f(past):
    global final_df
    df_list = []
    col_n = {'Close': 'DJI30'}
    x = pd.DataFrame(yf.download('^DJI', start='2003-12-01', end = today))
    x1 = x.reset_index()
    x2 = x1[['Date','Close']][-past:]
    x2.rename(columns = col_n, inplace=True)
    x2 = pd.DataFrame(x2.reset_index(drop=True)) 
    for label, name in comm_dict2.items(): 
        col_name = {'Close': name}
        y1 = pd.DataFrame(yf.download(label, start='2003-12-01', end = today)) 
        y1.reset_index()
        y2 = y1[['Close']][-past:]
        y2 = pd.DataFrame(y2.reset_index(drop=True))
        y2.rename(columns = col_name, inplace=True)
        m_tab = pd.concat([x2, y2], axis=1)
        df_list.append(m_tab)
        final_df = pd.concat(df_list, axis=1)
        final_df = final_df.T.drop_duplicates().T
        final_df.to_excel('Nm_data.xlsx')

@st.cache_data        
def rr_eur_pln():
    eur_df = pd.read_excel('Nm_data.xlsx')
    final_PLN_EUR = eur_df['EUR/PLN']
    f_df = eur_df[['DJI30', 'USD_EUR', 'USD/CNY', 'Crude_Oil', 'Gold', 'NASDAQ', 'SP_500',
           '10_YB', 'Copper', 'USD_GBP', 'USD_JPY', 'PLN/USD', '5_YB','USD/AED',
           'USD/RUB', 'Platinum', 'Silver', 'Natural Gas', 'Rice Futures',
           'Soy Futures', 'KC HRW Wheat Futures']]
    rr_d = (f_df - f_df.shift(1)) / f_df.shift(1)  # ta linijka kodu liczy wszystkie stopy zwrotu
    rr_df = pd.concat([final_PLN_EUR, rr_d], axis=1)
    rr_df.dropna()
    rr_df.to_excel('Nm_rr_eur.xlsx')

@st.cache_data    
def rr_usd_pln():
    usd_df = pd.read_excel('Nm_data.xlsx')
    final_PLN_USD = usd_df['PLN/USD']
    f_df1 = usd_df[['DJI30', 'USD_EUR', 'USD/CNY', 'Crude_Oil', 'Gold', 'NASDAQ', 'SP_500', 'EUR/PLN','USD/AED',
           '10_YB', 'Copper', 'USD_GBP', 'USD_JPY', '5_YB', 'USD/RUB', 'Platinum', 'Silver', 'Natural Gas',
            'Rice Futures','Soy Futures', 'KC HRW Wheat Futures']]
    rr_d1 = (f_df1 - f_df1.shift(1)) / f_df1.shift(1)  # ta linijka kodu liczy wszystkie stopy zwrotu
    rr_df1 = pd.concat([final_PLN_USD, rr_d1], axis=1)
    rr_df1.dropna()
    rr_df1.to_excel('Nm_rr_usd.xlsx')
        

# D+1 LSTM Prediction Model

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM

# Ostateczne definicje listych zmiennych

new_rr = pd.read_excel('Nm_rr_eur.xlsx')
n_rr_eur = new_rr[['EUR/PLN', 'DJI30', 'USD_EUR', 'USD/CNY', 'Crude_Oil',
            'Gold', 'NASDAQ', 'SP_500', '10_YB', 'Copper', 'USD_GBP', 'USD_JPY','USD/AED',
            'PLN/USD', '5_YB', 'USD/RUB', 'Platinum', 'Silver', 'Natural Gas',
            'Rice Futures', 'Soy Futures', 'KC HRW Wheat Futures']]
n_rr_eur.fillna(0)
n_rr_eur.to_excel('N_rr_eur.xlsx')
  
new_rr_usd = pd.read_excel('Nm_rr_usd.xlsx')
n_rr_usd = new_rr_usd[['PLN/USD','EUR/PLN','DJI30', 'USD_EUR', 'USD/CNY','Crude_Oil', 'Gold',
                        'NASDAQ', 'SP_500','10_YB', 'Copper', 'USD_GBP','USD_JPY','5_YB', 'USD/RUB','Platinum', 'USD/AED', 
                        'Silver', 'Natural Gas','Rice Futures','Soy Futures','KC HRW Wheat Futures']]
n_rr_usd.fillna(0)
n_rr_usd.to_excel('N_rr_usd.xlsx')

@st.cache_data
def LSTM_Model(data_set):
    set_1 = data_set.fillna(0)
    scaler=MinMaxScaler(feature_range=(0,1))
    set_1_scaled = scaler.fit_transform(np.array(set_1))
    tr_size=int(len(set_1_scaled)*0.7)
    te_size=len(set_1_scaled)-tr_size
    tr_data = set_1_scaled[0:tr_size,:]
    te_data = set_1_scaled[tr_size : tr_size+te_size,:]
    
    def create_dataset(dataset, time_step): # time_step=1
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):  # tutaj ustalamy na ile okresów w przód prognozujemy
            a = dataset[i:(i+time_step)]
            dataX.append(a)
            dataY.append(dataset[(i + time_step):(i + time_step+1), 0]) # tutaj ustalamy którą zmienną prognozujemy, oraz na ile okresów w przód
    
        return np.array(dataX), np.array(dataY)
    
    time_step = 100  # to jest krytyczny parametr
    X_train, y_train = create_dataset(tr_data, time_step)
    X_test, y_test = create_dataset(te_data, time_step)
    
    model = Sequential()
    model.add(LSTM(100,return_sequences=True,input_shape=(100,22))) # liczba kolumn musi tutaj być taka sama jak w X_train
    model.add(LSTM(100,return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(1)) # tutaj musi być wprowadzona liczba dni na które robimy prognozę
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size = 64) #, verbose = 1
    
    train_predi = model.predict(X_train) # , verbose=1
    test_predi = model.predict(X_test) # , verbose=1
    
    set150 = set_1[-100:].to_numpy() #set125 = np.array([set1[-125:].to_numpy()])
    set_150_sc = scaler.transform(set150) #set_125_sc.shape
    set_150_scaled = np.array([set_150_sc]) # set_125_scaled.shape
    
    forecast = model.predict(set_150_scaled) # , verbose=1
    _forecast = (forecast - scaler.min_[0])/scaler.scale_[0]
    _forecast_df = pd.DataFrame(_forecast)
    _forecast_df.to_excel('forecast.xlsx')
    
# Console definition

st.subheader(f'Today {today} we are loading following data: ', divider="red")
st.write(list(comm_dict2.values()))
    
if st.button('Data load',key = "<char1>"):
    yees = model_f(3001)    
    final_df = pd.read_excel('Nm_data.xlsx')
    st.subheader('Current data sample', divider="red")
    st.dataframe(final_df[['Date','EUR/PLN','PLN/USD','Crude_Oil']][-3:]) 
    
if st.button('Prepare Data for LSTM',key = "<char2>"):
    yees1 = rr_eur_pln()   
    st.subheader('EUR/PLN data sample', divider="red")
    eur_df = pd.read_excel('Nm_rr_eur.xlsx')
    st.dataframe(eur_df[['EUR/PLN','PLN/USD','Crude_Oil']][-3:])
    yees2 = rr_usd_pln()
    usd_df = pd.read_excel('Nm_rr_usd.xlsx')
    st.subheader('USD/PLN data sample', divider="red")
    st.dataframe(usd_df[['PLN/USD','EUR/PLN','Crude_Oil']][-3:])   

st.subheader('EUR/PLN or USD/PLN ', divider="red")

col1, col2 = st.columns(2)

with col1:
    checkbox_eur = st.checkbox('EUR/PLN Data set',key = "<lstm1>")    
    
    if checkbox_eur:
        LSTM_Model(n_rr_eur)
        _forecast_eur = pd.read_excel('forecast.xlsx')
        st.subheader(f'EUR/PLN prediction for today {today} is {list(_forecast_eur[0])}', divider="blue")
        _forecast_eur.to_excel('forecast_eur.xlsx')

with col2:
    checkbox_usd = st.checkbox('USD/PLN Data set',key = "<lstm2>")    
    
    if checkbox_usd:
        LSTM_Model(n_rr_usd)
        _forecast_usd = pd.read_excel('forecast.xlsx')
        st.subheader(f'USD/PLN prediction for today is {list(_forecast_usd[0])}', divider="blue")
        _forecast_usd.to_excel('forecast_usd.xlsx')

