import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np


def zone_and_linked(ax, axins, zone_left, zone_right, x, y, linked='bottom',
                    x_ratio=0.05, y_ratio=0.05):
    """缩放内嵌图形，并且进行连线
    ax:         调用plt.subplots返回的画布。例如： fig,ax = plt.subplots(1,1)
    axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    zone_left:  要放大区域的横坐标左端点
    zone_right: 要放大区域的横坐标右端点
    x:          X轴标签
    y:          列表，所有y值
    linked:     进行连线的位置，{'bottom','top','left','right'}
    x_ratio:    X轴缩放比例
    y_ratio:    Y轴缩放比例
    """
    y_ratio = 0.7
    xlim_left = x[zone_left] - (x[zone_right] - x[zone_left]) * x_ratio
    xlim_right = x[zone_right] + (x[zone_right] - x[zone_left]) * x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data) - (np.max(y_data) - np.min(y_data)) * y_ratio
    ylim_top = np.max(y_data) + (np.max(y_data) - np.min(y_data)) * y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left, xlim_right, xlim_right, xlim_left, xlim_left],
            [ylim_bottom, ylim_bottom, ylim_top, ylim_top, ylim_bottom], "red")

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_left, ylim_bottom)
        xyA_2, xyB_2 = (xlim_right, ylim_top), (xlim_right, ylim_bottom)
    elif linked == 'top':
        xyA_1, xyB_1 = (xlim_left, ylim_bottom), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_right, ylim_top)
    elif linked == 'left':
        xyA_1, xyB_1 = (xlim_right, ylim_top), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_left, ylim_bottom)
    elif linked == 'right':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_right, ylim_top)
        xyA_2, xyB_2 = (xlim_left, ylim_bottom), (xlim_right, ylim_bottom)

    con = ConnectionPatch(xyA=xyA_1, xyB=xyB_1, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax, color='red')
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2, xyB=xyB_2, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax, color='red')
    axins.add_artist(con)


pre=pd.read_excel('第四组 S07.xlsx')

pre=pre.iloc[:,1:].reset_index()
pre.rename(columns={'True':'Measured'},inplace=True)

y1=pre['Proposed Model']
y2=pre['LSTM']
y3=pre['XGBoost']
y4=pre['Random Forest']
y5=pre['SVR']
y6=pre['Measured']

fig, ax = plt.subplots(1,1,figsize=(13,6),dpi=600)
# ax.grid()
# fig = plt.figure(dpi=600)
plt.grid()
plt.plot(y1,color='#669BBB',label='Proposed Model',marker='^',markerfacecolor='none')
plt.plot(y2,color='#F66F69',label='LSTM',marker='.',markerfacecolor='none')
plt.plot(y3,color='#B7C88C',label='XGBoost',marker='s',markerfacecolor='none')
plt.plot(y4,color='#023047',label='Random Forest',marker='v',markerfacecolor='none')
plt.plot(y5,color='#9A8CB4',label='SVR',marker='H',markerfacecolor='none')
plt.plot(y6,color='black',label='Measured',marker='*',markerfacecolor='none')
axins = ax.inset_axes((1.1, 0.2, 0.3, 0.6))
axins.grid()
axins.plot(y1,color='#669BBB',label='Proposed Model',marker='^',markerfacecolor='none')
axins.plot(y2,color='#F66F69',label='LSTM',marker='.',markerfacecolor='none')
axins.plot(y3,color='#B7C88C',label='XGBoost',marker='s',markerfacecolor='none')
axins.plot(y4,color='#023047',label='Random Forest',marker='v',markerfacecolor='none')
axins.plot(y5,color='#9A8CB4',label='SVR',marker='H',markerfacecolor='none')
axins.plot(y6,color='black',label='Measured',marker='*',markerfacecolor='none')
zone_and_linked(ax, axins, 16, 18, pre['index'] , [pre[i] for i in pre.columns[1:]], 'right')
plt.vlines(16,np.min(y6),np.max(y6),linestyles='dashed',linewidths=2)
plt.xticks(np.arange(0,18,2))
plt.ylabel('Tunnel settlement (mm)')
plt.xlabel('Time (Semi-annual)')
plt.legend(bbox_to_anchor=(1.1,-0.1),ncol=6)
plt.savefig('./predictre/4S07.png',dpi=600,bbox_inches = 'tight')
plt.show()

