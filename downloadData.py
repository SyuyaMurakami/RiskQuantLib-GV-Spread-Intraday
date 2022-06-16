#!/usr/bin/python
# coding = utf-8
import os
import sys

import pandas as pd

from RiskQuantLib.Module import *
from RiskQuantLib.Tool.fileTool import loadVariable,dumpVariable
path = sys.path[0]

def downloadData():
    from RiskQuantLib.DataInputAPI.getDataFromWind import getOptionMarketData,getAllStockOfSector,getSHIBOR
    from RiskQuantLib.Tool.strTool import getLastTradingDate,generateTradingDateList,changeSecurityListToStr
    today = pd.Timestamp.now()
    lastTradingDay = getLastTradingDate(today)
    # Update Option Data
    df = loadVariable(path+os.sep+"options.pkl")
    latestUpdateDate = pd.Timestamp(max(df['DATE']))
    if today.strftime("%Y%m%d") != latestUpdateDate.strftime("%Y%m%d"):
        callSectorCode = '1000018861000000'
        putSectorCode = '1000018862000000'

        callOptionCode = getAllStockOfSector(today, callSectorCode)
        putOptionCode = getAllStockOfSector(today, putSectorCode)
        optionCode = callOptionCode['wind_code'].to_list() + putOptionCode['wind_code'].to_list()
        dayInfo = getOptionMarketData(changeSecurityListToStr(optionCode), today)

        if not dayInfo.empty:
            tmp = dayInfo
            tmp['DATE'] = [i.strftime("%Y/%m/%d") for i in tmp['DATE']]
            dumpVariable(tmp, path+os.sep+"options.pkl")
            print("Option Data Download Finished")
    else:
        print("Options Already Updated")

    # Update SHIBOR
    df = loadVariable(path+os.sep+'shibor.pkl')
    if df.shape[1]==1 and df.columns[0]==lastTradingDay.strftime("%Y-%m-%d"):
        print("SHIBOR Already Updated")
    else:
        tmp = getSHIBOR(lastTradingDay)
        df = pd.DataFrame(tmp.values,columns=[lastTradingDay.strftime("%Y-%m-%d")], index=['1D','1W','2W','1M','3M','6M','9M','1Y'])
    dumpVariable(df,path+os.sep+'shibor.pkl')

    print("Download Data Finish")