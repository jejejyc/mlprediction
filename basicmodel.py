import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
#导入评价指标
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import r2_score as r2
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

def XGB(x_train, y_train):
    my_cv = TimeSeriesSplit(n_splits=2).split(x_train)# 限定max_train_size
    cv_params = {'n_estimators': [50, 150,200], 'learning_rate': [0.01, 0.1, 0.3, 1], 'max_depth': [2, 3,4,5],
                 'min_child_weight': [4, 5, 6, 7, 8], 'gamma': [1, 3], 'reg_alpha': [0.1, 0.3]}
#     other_params = {'learning_rate': 0.1, 'n_estimators': 90, 'max_depth': 7, 'min_child_weight': 4, 'seed': 0,
#                     'subsample': 1, 'colsample_bytree': 0.9, 'gamma': 1, 'reg_alpha': 0.1, "lambda": 0.9}
    model = XGBRegressor()
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_absolute_error', cv=my_cv)
    optimized_GBM.fit(np.array(x_train), np.array(y_train))
    model = optimized_GBM.best_estimator_
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    return model

def LGB(x_train, y_train):
    my_cv = TimeSeriesSplit(n_splits=2).split(x_train)# 限定max_train_size
    cv_params ={
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'num_leaves': [10, 20, 30]
}
    model = LGBMRegressor()
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_absolute_error', cv=my_cv)
    optimized_GBM.fit(np.array(x_train), np.array(y_train))
    model = optimized_GBM.best_estimator_
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    return model

def svr(x_train, y_train):
    my_cv = TimeSeriesSplit(n_splits=2).split(x_train)# 限定max_train_size
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf']
    }
    model =SVR()
    optimized_GBM = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=my_cv)
    optimized_GBM.fit(np.array(x_train), np.array(y_train))
    model = optimized_GBM.best_estimator_
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    return model

def rf(x_train, y_train):
    my_cv = TimeSeriesSplit(n_splits=2).split(x_train)# 限定max_train_size
    param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5],
    'max_features': ['auto', 'sqrt', 'log2']
    }
    model = RandomForestRegressor()
    optimized_GBM = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=my_cv)
    optimized_GBM.fit(np.array(x_train), np.array(y_train))
    model = optimized_GBM.best_estimator_
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    return model

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

'''
MAPE
'''
#训练集
xgbmape=[]
lgbmape=[]
svrmape=[]
rfmape=[]
#测试集
txgbmape=[]
tlgbmape=[]
tsvrmape=[]
trfmape=[]
#总体
axgbmape=[]
algbmape=[]
asvrmape=[]
arfmape=[]


'''
MSE
'''
#训练集
xgbmse=[]
lgbmse=[]
svrmse=[]
rfmse=[]
#测试集
txgbmse=[]
tlgbmse=[]
tsvrmse=[]
trfmse=[]
#总体
axgbmse=[]
algbmse=[]
asvrmse=[]
arfmse=[]



'''
MAE
'''
#训练集
xgbmae=[]
lgbmae=[]
svrmae=[]
rfmae=[]
#测试集
txgbmae=[]
tlgbmae=[]
tsvrmae=[]
trfmae=[]
#总体
axgbmae=[]
algbmae=[]
asvrmae=[]
arfmae=[]



'''
R2
'''
#训练集
xgbmr2=[]
lgbmr2=[]
svrmr2=[]
rfmr2=[]
#测试集
txgbmr2=[]
tlgbmr2=[]
tsvrmr2=[]
trfmr2=[]
#总体
axgbmr2=[]
algbmr2=[]
asvrmr2=[]
arfmr2=[]


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
    #xgb
    # model1=XGB(X_train,Y_train)
    # #lgb
    # model2=LGB(X_train,Y_train)
    #svr
    model3=svr(X_train,Y_train)
    #rf
    # model4=rf(X_train,Y_train)


    '''
    MAPE
    '''
    #训练集
    # xgbmape.append(MAPE(Y_train,model1.predict(X_train)))
    # lgbmape.append(MAPE(Y_train,model2.predict(X_train)))
    svrmape.append(MAPE(Y_train,model3.predict(X_train)))
    # rfmape.append(MAPE(Y_train,model4.predict(X_train)))
    #测试集
    # txgbmape.append(MAPE(Y_test, model1.predict(X_test)))
    # tlgbmape.append(MAPE(Y_test, model2.predict(X_test)))
    tsvrmape.append(MAPE(Y_test, model3.predict(X_test)))
    # trfmape.append(MAPE(Y_test, model4.predict(X_test)))
    #总体
    # axgbmape.append(MAPE(Y, model1.predict(X)))
    # algbmape.append(MAPE(Y, model2.predict(X)))
    asvrmape.append(MAPE(Y, model3.predict(X)))
    # arfmape.append(MAPE(Y, model4.predict(X)))

    '''
    MSE
    '''
    #训练集
    # xgbmse.append(MSE(Y_train,model1.predict(X_train)))
    # lgbmse.append(MSE(Y_train,model2.predict(X_train)))
    svrmse.append(MSE(Y_train,model3.predict(X_train)))
    # rfmse.append(MSE(Y_train,model4.predict(X_train)))
    #测试集
    # txgbmse.append(MSE(Y_test, model1.predict(X_test)))
    # tlgbmse.append(MSE(Y_test, model2.predict(X_test)))
    tsvrmse.append(MSE(Y_test, model3.predict(X_test)))
    # trfmse.append(MSE(Y_test, model4.predict(X_test)))
    #总体
    # axgbmse.append(MSE(Y, model1.predict(X)))
    # algbmse.append(MSE(Y, model2.predict(X)))
    asvrmse.append(MSE(Y, model3.predict(X)))
    # arfmse.append(MSE(Y, model4.predict(X)))

    '''
    MAE
    '''
    # 训练集
    # xgbmae.append(MAE(Y_train, model1.predict(X_train)))
    # lgbmae.append(MAE(Y_train, model2.predict(X_train)))
    svrmae.append(MAE(Y_train, model3.predict(X_train)))
    # rfmae.append(MAE(Y_train, model4.predict(X_train)))
    # 测试集
    # txgbmae.append(MAE(Y_test, model1.predict(X_test)))
    # tlgbmae.append(MAE(Y_test, model2.predict(X_test)))
    tsvrmae.append(MAE(Y_test, model3.predict(X_test)))
    # trfmae.append(MAE(Y_test, model4.predict(X_test)))
    # 总体
    # axgbmae.append(MAE(Y, model1.predict(X)))
    # algbmae.append(MAE(Y, model2.predict(X)))
    asvrmae.append(MAE(Y, model3.predict(X)))
    # arfmae.append(MAE(Y, model4.predict(X)))

    '''
    R2
    '''
    # 训练集
    # xgbmr2.append(r2(Y_train, model1.predict(X_train)))
    # lgbmr2.append(r2(Y_train, model2.predict(X_train)))
    svrmr2.append(r2(Y_train, model3.predict(X_train)))
    # rfmr2.append(r2(Y_train, model4.predict(X_train)))
    # 测试集
    # txgbmr2.append(r2(Y_test, model1.predict(X_test)))
    # tlgbmr2.append(r2(Y_test, model2.predict(X_test)))
    tsvrmr2.append(r2(Y_test, model3.predict(X_test)))
    # trfmr2.append(r2(Y_test, model4.predict(X_test)))
    # 总体
    # axgbmr2.append(r2(Y, model1.predict(X)))
    # algbmr2.append(r2(Y, model2.predict(X)))
    asvrmr2.append(r2(Y, model3.predict(X)))
    # arfmr2.append(r2(Y, model4.predict(X)))
#MAPE
dmape=pd.DataFrame()
dmape['points']=points
# dmape['xgb']=xgbmape
# dmape['lgb']=lgbmape
# dmape['rf']=rfmape
dmape['svr']=svrmape
tdmape=pd.DataFrame()
tdmape['points']=points
# tdmape['xgb']=txgbmape
# tdmape['lgb']=tlgbmape
# tdmape['rf']=trfmape
tdmape['svr']=tsvrmape
admape=pd.DataFrame()
admape['points']=points
# admape['xgb']=axgbmape
# admape['lgb']=algbmape
admape['svr']=asvrmape
# admape['rf']=arfmape
writer = pd.ExcelWriter('svrmape.xlsx')
dmape.to_excel(writer,"train")
tdmape.to_excel(writer,"test")
admape.to_excel(writer,'all')
writer.save()
writer.close()

#MSE
dmse=pd.DataFrame()
dmse['points']=points
# dmse['xgb']=xgbmse
# dmse['lgb']=lgbmse
# dmse['rf']=rfmse
dmse['svr']=svrmse
tdmse=pd.DataFrame()
tdmse['points']=points
tdmse['svr']=tsvrmse
# tdmse['xgb']=txgbmse
# tdmse['lgb']=tlgbmse
# tdmse['rf']=trfmse
admse=pd.DataFrame()
admse['points']=points
admse['svr']=asvrmse
# admse['xgb']=axgbmse
# admse['lgb']=algbmse
# admse['rf']=arfmse
writer = pd.ExcelWriter('svrmse.xlsx')
dmse.to_excel(writer,"train")
tdmse.to_excel(writer,"test")
admse.to_excel(writer,'all')
writer.save()
writer.close()



#MAE
dmae=pd.DataFrame()
dmae['points']=points
dmae['svr']=svrmae
# dmae['xgb']=xgbmae
# dmae['lgb']=lgbmae
# dmae['rf']=rfmae
tdmae=pd.DataFrame()
tdmae['points']=points
tdmae['svr']=tsvrmae
# tdmae['xgb']=txgbmae
# tdmae['lgb']=tlgbmae
# tdmae['rf']=trfmae
admae=pd.DataFrame()
admae['points']=points
admae['svr']=asvrmae
# admae['xgb']=axgbmae
# admae['lgb']=algbmae
# admae['rf']=arfmae
writer = pd.ExcelWriter('svrmae.xlsx')
dmae.to_excel(writer,"train")
tdmae.to_excel(writer,"test")
admae.to_excel(writer,'all')
writer.save()
writer.close()

#R2
dmr2=pd.DataFrame()
dmr2['points']=points
dmr2['svr']=svrmr2
# dmr2['xgb']=xgbmr2
# dmr2['lgb']=lgbmr2
# dmr2['rf']=rfmr2
tdmr2=pd.DataFrame()
tdmr2['points']=points
tdmr2['svr']=tsvrmr2
# tdmr2['xgb']=txgbmr2
# tdmr2['lgb']=tlgbmr2
# tdmr2['rf']=trfmr2
admr2=pd.DataFrame()
admr2['points']=points
admr2['svr']=asvrmr2
# admr2['xgb']=axgbmr2
# admr2['lgb']=algbmr2
# admr2['rf']=arfmr2
writer = pd.ExcelWriter('svrr2.xlsx')
dmr2.to_excel(writer,"train")
tdmr2.to_excel(writer,"test")
admr2.to_excel(writer,'all')
writer.save()
writer.close()
