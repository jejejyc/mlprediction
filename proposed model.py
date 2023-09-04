import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split,GridSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt
#导入评价指标
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import r2_score as r2
from PyEMD import CEEMDAN
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from collections import Counter


df=pd.read_excel('file.xlsx')
'''
改
'''
group='分组4'
g=df[df['group']==4]
points=g.date.values
g=g.iloc[:,1:-1]
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

#统一构建训练数据
def traindata(x):
    t=[]
    for i in range(x.shape[0]):
        values = x[i, :]
        '''
        窗口大小
        '''
        data = series_to_supervised(values, n_in=3)
        t.append(data)
    t=np.array(t)
    return t

#贝叶斯优化预测器   name参数保存的名字（X_train,y_train,g1）
def bay_build_model(X_train,y_train,name):
    def xgb_cv(max_depth, learning_rate, n_estimators, min_child_weight, subsample, colsample_bytree, reg_alpha, gamma):
        val = cross_val_score(estimator=xgb.XGBRegressor(max_depth=int(max_depth),
                                                     learning_rate=learning_rate,
                                                     n_estimators=int(n_estimators),
                                                     min_child_weight=min_child_weight,
                                                     subsample=max(min(subsample, 1), 0),
                                                     colsample_bytree=max(min(colsample_bytree, 1), 0),
                                                     reg_alpha=max(reg_alpha, 0), gamma=gamma,
                                                     # objective='reg:squarederror',
                                                     # booster='gbtree',
                                                     seed=888), X=X_train, y=y_train, scoring='neg_mean_absolute_error',
                              cv=5).mean()
        return val

    xgb_bo = BayesianOptimization(xgb_cv, pbounds={'max_depth': (1, 10),
                                                   'learning_rate': (0.01, 0.3),
                                                   'n_estimators': (1, 1000),
                                                   'min_child_weight': (0, 20),
                                                   'subsample': (0.001, 1),
                                                   'colsample_bytree': (0.01, 1),
                                                   'reg_alpha': (0.001, 20),
                                                   'gamma': (0.001, 10)})
    xgb_bo.maximize(n_iter=100, init_points=10)
    pa = pd.DataFrame.from_dict(xgb_bo.max)
    pa.to_excel('%s优化后参数取值.xlsx'%name)
    params = xgb_bo.max['params']
    xgb1 = xgb.XGBRegressor(gamma=params['gamma'], colsample_bytree=params['colsample_bytree'],
                        learning_rate=params['learning_rate'],
                        max_depth=int(params['max_depth']), min_child_weight=params['min_child_weight'],
                        n_estimators=int(params['n_estimators']),
                        reg_alpha=params['reg_alpha'], subsample=params['subsample'],
                        objective='reg:squarederror',
                        booster='gbtree',
                        n_jobs=4)


    # 模型训练

    grid_result = xgb1.fit(X_train, y_train)

    return grid_result

def train_model(allimfs,n_imf,name):
    imfs=allimfs[:,n_imf,:]
    dt=traindata(imfs)
    ntrain=int(dt.shape[1]*0.8)
    X_train=dt[:,:ntrain,:-1].reshape(-1,3)
    y_train=dt[:,:ntrain,-1].reshape(-1,1)
    model=bay_build_model(X_train,y_train,name)
    return model

def evaluation_model(models,allimfs,df,points,name):
    #训练集
    df=np.array(df)
    tt=traindata(df)
    # 训练集
    mape,mae,mse,mr2=[],[],[],[]
    #测试集
    tmape, tmae, tmse, tmr2 = [], [], [], []
    # 全部
    amape, amae, amse, amr2 = [], [], [], []
    #tt.shape:[测点编号，样本量，总特征数+预测值]
    for i in range(tt.shape[0]):
        print('训练到%d测点'%(i+1))
        y_true=tt[i,:,-1]
        imfs=allimfs[i,:,:]
        # 分量1
        imfs1 = traindata(imfs[0, :].reshape(1, imfs.shape[1]))
        imfs1 = imfs1.reshape(-1,4)
        # 分量2
        imfs2 = traindata(imfs[1, :].reshape(1, imfs.shape[1]))
        imfs2 = imfs2.reshape(-1, 4)
        # 分量3
        imfs3 = traindata(imfs[2, :].reshape(1, imfs.shape[1]))
        imfs3 = imfs3.reshape(-1, 4)
        nt=int(len(imfs1)*0.8)
        #测试集
        im1y = models[0].predict(imfs1[:nt,:-1])
        im2y = models[1].predict(imfs2[:nt,:-1])
        im3y = models[2].predict(imfs3[:nt,:-1])
        pre=im1y+im2y+im3y
        mape.append(MAPE(y_true[:nt],pre))
        mae.append(MAE(y_true[:nt],pre))
        mse.append(MSE(y_true[:nt],pre))
        mr2.append(r2(y_true[:nt],pre))
        # 训练集
        im1y = models[0].predict(imfs1[nt:, :-1])
        im2y = models[1].predict(imfs2[nt:, :-1])
        im3y = models[2].predict(imfs3[nt:, :-1])
        pre = im1y + im2y + im3y
        tmape.append(MAPE(y_true[nt:], pre))
        tmae.append(MAE(y_true[nt:], pre))
        tmse.append(MSE(y_true[nt:], pre))
        tmr2.append(r2(y_true[nt:], pre))
        # 全部
        im1y = models[0].predict(imfs1[:, :-1])
        im2y = models[1].predict(imfs2[:, :-1])
        im3y = models[2].predict(imfs3[:, :-1])
        pre = im1y + im2y + im3y
        amape.append(MAPE(y_true, pre))
        amae.append(MAE(y_true, pre))
        amse.append(MSE(y_true, pre))
        amr2.append(r2(y_true, pre))

    # MAPE
    dmape = pd.DataFrame()
    dmape['points'] = points
    dmape['train'] = mape
    dmape['test'] = tmape
    dmape['all'] = amape

    # MSE
    dmse = pd.DataFrame()
    dmse['points'] = points
    dmse['train'] = mse
    dmse['test'] = tmse
    dmse['all'] = amse

    # MAE
    dmae = pd.DataFrame()
    dmae['points'] = points
    dmae['train'] = mae
    dmae['test'] = tmae
    dmae['all'] = amae

    # MR2
    dmr2 = pd.DataFrame()
    dmr2['points'] = points
    dmr2['train'] = mr2
    dmr2['test'] = tmr2
    dmr2['all'] = amr2

    # writer = pd.ExcelWriter('Proposed model_%s.xlsx'%name)
    # dmape.to_excel(writer, 'mape')
    # dmse.to_excel(writer, 'mse')
    # dmae.to_excel(writer, 'mae')
    # dmr2.to_excel(writer, 'mr2')
    # writer.save()
    # writer.close()


#模态分解 最大为2
allimfs=[]
p=1
for i ,row in g.iterrows():
    print('分解到第%d个测点'%p)
    ts=np.array(row)
#规定最大分量数目为2
    ceemdan = CEEMDAN()(ts,max_imf=2)
    allimfs.append(ceemdan)
    p+=1
allimfs=np.array(allimfs)
print('所有分量数',allimfs.shape)



print('*************训练第一分量************')
model1=train_model(allimfs,0,'%s分量1'%group)
print('*************训练第二分量************')
model2=train_model(allimfs,1,'%s分量2'%group)
print('*************训练第三分量************')
model3=train_model(allimfs,2,'%s分量3'%group)
models=[model1,model2,model3]

print('*************模型评价**************')
evaluation_model(models,allimfs,g,points,group)




