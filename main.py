#!/usr/bin/python
# coding = utf-8
import os
import sys
from WindPy import w
w.start(waitTime=60)
import numpy as np
import pandas as pd
import warnings
from pyecharts import Line

from RiskQuantLib.Module import *
from RiskQuantLib.Tool.fileTool import loadVariable,dumpVariable
from RiskQuantLib.Tool.strTool import changeSecurityListToStr,getLastTradingDate
from downloadData import downloadData
from iVIX import calDayVIX
path = sys.path[0]

# collect file of last trading day
today  = pd.Timestamp.now()
lastTradingDay = getLastTradingDate(today)
targetPath = path+os.sep+"History"+os.sep+lastTradingDay.strftime("%Y%m%d")
if os.path.exists(targetPath):
    pass
else:
    os.mkdir(targetPath)
    os.system('''MOVE "'''+path+os.sep+'vix.html'+'''" "'''+targetPath+os.sep+"vix.html"+'''"''')
downloadData()

options = loadVariable(path+os.sep+"options.pkl")
shibor = loadVariable(path+os.sep+"shibor.pkl")
closeDict = dict(zip(options.index,[np.nan for i in range(options.shape[0])]))
gvSpread = []
timeStamp = []

# initialize price
initData = w.wsq(changeSecurityListToStr(options.index), "rt_last", usedf=True)
closeDict.update(dict(zip(initData[1].index,initData[1]['RT_LAST'])))

# intra-day callback
def callBack(windData):
    updateDict = dict(zip(windData.Codes,windData.Data[0]))
    closeDict.update(updateDict)
    tmp = options[['SEC_NAME','EXE_MODE','EXE_PRICE','EXE_ENDDATE']].copy(deep=True)
    tmp['CLOSE'] = [closeDict[i] for i in tmp.index]
    tmp.index = [today.strftime("%Y/%m/%d") for i in tmp.index]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        vix,gvix = calDayVIX(today.strftime("%Y/%m/%d"),tmp.fillna(0),shibor.T)
        gvSpread.append(gvix-vix)
        timeStamp.append(windData.Times[0].strftime("%Y/%m/%d %H:%M:%S"))
    line = Line(u'GV Spread')
    line.add("Intra-Day Volatility", timeStamp,gvSpread,mark_line=['max','average'])
    line.render('vix.html')


w.wsq(changeSecurityListToStr(options.index), "rt_last", func=callBack)
q = ''
while q!='c':
    q = input()
w.cancelRequest(0)
print("GV_Spread Finish")