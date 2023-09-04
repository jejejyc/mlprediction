#导入评价指标
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import r2_score as r2
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#构建数据集
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	df = pd.DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = pd.concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values
df=pd.read_excel('file.xlsx')
points=df.date.values

def LSTM_Model(X_train, Y_train):
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, Y_train, epochs=100, batch_size=1, validation_split=0.1, verbose=1, shuffle=True)
    return (model)


#MAPE
mape=[]
tmape=[]
amape=[]
#MSE
mse=[]
tmse=[]
amse=[]

#MAE
mae=[]
tmae=[]
amae=[]


#MR2
mr2=[]
tmr2=[]
amr2=[]

for i in range(df.shape[0]):
    print(df.iloc[i,0])
    t=df.iloc[i,1:-1]
    t1=np.array(t).reshape(-1,1)
    data=series_to_supervised(t1, n_in=3, n_out=1, dropnan=True)
    X=data[:,:-1]
    Y=data[:,-1]
    ntrain=int(len(X)*0.8)
    X_train=X[:ntrain]
    Y_train=Y[:ntrain]
    X_test=X[ntrain:]
    Y_test=Y[ntrain:]
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    model=LSTM_Model(X_train,Y_train)

    # MAPE
    mape.append(MAPE(Y_train,model.predict(X_train)))
    tmape.append(MAPE(Y_test,model.predict(X_test)))
    amape.append(MAPE(Y,model.predict(X)))
 # MSE
    mse.append(MSE(Y_train,model.predict(X_train)))
    tmse.append(MSE(Y_test,model.predict(X_test)))
    amse.append(MSE(Y,model.predict(X)))
    # MAE
    mae.append(MAE(Y_train, model.predict(X_train)))
    tmae.append(MAE(Y_test, model.predict(X_test)))
    amae.append(MAE(Y, model.predict(X)))
    # MR2
    mr2.append(r2(Y_train, model.predict(X_train)))
    tmr2.append(r2(Y_test, model.predict(X_test)))
    amr2.append(r2(Y, model.predict(X)))




#MAPE
dmape=pd.DataFrame()
dmape['points']=points
dmape['train']=mape
dmape['test']=tmape
dmape['all']=amape

#MSE
dmse=pd.DataFrame()
dmse['points']=points
dmse['train']=mse
dmse['test']=tmse
dmse['all']=amse

#MAE
dmae=pd.DataFrame()
dmae['points']=points
dmae['train']=mae
dmae['test']=tmae
dmae['all']=amae


#MR2
dmr2=pd.DataFrame()
dmr2['points']=points
dmr2['train']=mr2
dmr2['test']=tmr2
dmr2['all']=amr2

writer = pd.ExcelWriter('LSTM.xlsx')
dmape.to_excel(writer,'mape')
dmse.to_excel(writer,'mse')
dmae.to_excel(writer,'mae')
dmr2.to_excel(writer,'mr2')
writer.save()
writer.close()
