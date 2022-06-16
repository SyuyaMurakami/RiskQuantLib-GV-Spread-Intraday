# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 09:52:52 2018

@author: 量小白
"""
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.interpolate import make_interp_spline
import sys,os
from RiskQuantLib.Tool.strTool import isTradingDate,getLastTradingDate

#==============================================================================
# 开始计算ivix部分
#==============================================================================
def periodsSplineRiskFreeInterestRate(options, date, shibor_rate):
    """
    params: options: 计算VIX的当天的options数据用来获取expDate
            date: 计算哪天的VIX
    return：shibor：该date到每个到期日exoDate的risk free rate

    """
    date = datetime.strptime(date,'%Y/%m/%d')
    #date = datetime(date.year,date.month,date.day)
    exp_dates = np.sort(options.EXE_ENDDATE.unique())
    periods = {}
    for epd in exp_dates:
        epd = pd.to_datetime(epd)
        periods[epd] = (epd - date).days*1.0/365.0
    shibor_date = datetime.strptime(shibor_rate.index[0], "%Y-%m-%d") 
    if date >= shibor_date:
        date_str = shibor_rate.index[0]
        shibor_values = shibor_rate.iloc[0].values
        #shibor_values = np.asarray(list(map(float,shibor_values)))
    else:
        date_str = date.strftime("%Y-%m-%d")
        if date_str in shibor_rate.index:
            shibor_values = shibor_rate.loc[date_str].values
        else:
            date_str = getLastTradingDate(date_str).strftime("%Y-%m-%d")
            shibor_values = shibor_rate.loc[date_str].values
        #shibor_values = np.asarray(list(map(float,shibor_values)))
        
    shibor = {}
    period = np.asarray([1.0, 7.0, 14.0, 30.0, 90.0, 180.0, 270.0, 360.0]) / 360.0
    min_period = min(period)
    max_period = max(period)
    for p in periods.keys():
        tmp = periods[p]
        if periods[p] > max_period:
            tmp = max_period * 0.99999
        elif periods[p] < min_period:
            tmp = min_period * 1.00001
        # 此处使用SHIBOR来插值
        sh = make_interp_spline(period, shibor_values, 3)
        shibor[p] = sh(tmp)/100.0
    return shibor


def getHistDayOptions(vixDate,options_data):
    options_data = options_data.loc[vixDate,:]
    return options_data
    
    

def getNearNextOptExpDate(options, vixDate):
    # 找到options中的当月和次月期权到期日；
    # 用这两个期权隐含的未来波动率来插值计算未来30隐含波动率，是为市场恐慌指数VIX；
    # 如果options中的最近到期期权离到期日仅剩1天以内，则抛弃这一期权，改
    # 选择次月期权和次月期权之后第一个到期的期权来计算。
    # 返回的near和next就是用来计算VIX的两个期权的到期日
    """
    params: options: 该date为交易日的所有期权合约的基本信息和价格信息
            vixDate: VIX的计算日期
    return: near: 当月合约到期日（ps：大于1天到期）
            next：次月合约到期日
    """
    vixDate = datetime.strptime(vixDate,'%Y/%m/%d')
    optionsExpDate = list(pd.Series(options.EXE_ENDDATE.values.ravel()).unique())
    optionsExpDate = [datetime.strptime(i,'%Y/%m/%d %H:%M') for i in optionsExpDate]
    near = min(optionsExpDate)
    optionsExpDate.remove(near)
    if (near - vixDate).days < 1 and len(optionsExpDate)>1:
        near = min(optionsExpDate)
        optionsExpDate.remove(near)
    if len(optionsExpDate)==0:
        nt = near
    else:
        nt = min(optionsExpDate)
    return near, nt

def getStrikeMinCallMinusPutClosePrice(options):
    # options 中包括计算某日VIX的call和put两种期权，
    # 对每个行权价，计算相应的call和put的价格差的绝对值，
    # 返回这一价格差的绝对值最小的那个行权价，
    # 并返回该行权价对应的call和put期权价格的差
    """
    params:options: 该date为交易日的所有期权合约的基本信息和价格信息
    return: strike: 看涨合约价格-看跌合约价格 的差值的绝对值最小的行权价
            priceDiff: 以及这个差值，这个是用来确定中间行权价的第一步
    """
    call = options[options.EXE_MODE==u"认购"].set_index(u"EXE_PRICE").sort_index()
    put  = options[options.EXE_MODE==u"认沽"].set_index(u"EXE_PRICE").sort_index()
    callMinusPut = call.CLOSE - put.CLOSE
    strike = abs(callMinusPut).idxmin()
    priceDiff = callMinusPut[strike].min()
    return strike, priceDiff

def calSigmaSquare(options, FF, R, T):
    # 计算某个到期日期权对于VIX的贡献sigma；
    # 输入为期权数据options，FF为forward index price，
    # R为无风险利率， T为期权剩余到期时间
    """
    params: options:该date为交易日的所有期权合约的基本信息和价格信息
            FF: 根据上一步计算得来的strike，然后再计算得到的forward index price， 根据它对所需要的看涨看跌合约进行划分。
                取小于FF的第一个行权价为中间行权价K0， 然后选取大于等于K0的所有看涨合约， 选取小于等于K0的所有看跌合约。
                对行权价为K0的看涨看跌合约，删除看涨合约，不过看跌合约的价格为两者的均值。
            R： 这部分期权合约到期日对应的无风险利率 shibor
            T： 还有多久到期（年化）
    return：Sigma：得到的结果是传入该到期日数据的Sigma
    """
    callAll = options[options.EXE_MODE==u"认购"].set_index(u"EXE_PRICE").sort_index()
    putAll  = options[options.EXE_MODE==u"认沽"].set_index(u"EXE_PRICE").sort_index()
    callAll['deltaK'] = 0.05
    putAll['deltaK']  = 0.05
    
    # Interval between strike prices
    index = callAll.index
    if len(index) < 3:
        callAll['deltaK'] = index[-1] - index[0]
    else:
        for i in range(1,len(index)-1):
            callAll['deltaK'].loc[index[i]] = (index[i+1]-index[i-1])/2.0
        callAll['deltaK'].loc[index[0]] = index[1]-index[0]
        callAll['deltaK'].loc[index[-1]] = index[-1] - index[-2]
    index = putAll.index
    if len(index) < 3:
        putAll['deltaK'] = index[-1] - index[0]
    else:
        for i in range(1,len(index)-1):
            putAll['deltaK'].loc[index[i]] = (index[i+1]-index[i-1])/2.0
        putAll['deltaK'].loc[index[0]] = index[1]-index[0]
        putAll['deltaK'].loc[index[-1]] = index[-1] - index[-2]
    
    call = callAll[callAll.index > FF]
    put  = putAll[putAll.index < FF]
    FF_idx = FF
    if put.empty:
        FF_idx = call.index[0]
        callComponent = call.CLOSE*call.deltaK/call.index/call.index
        sigma = (sum(callComponent))*np.exp(T*R)*2/T
        sigma = sigma - (FF/FF_idx - 1)**2/T
    elif call.empty:
        FF_idx = put.index[-1]
        putComponent = put.CLOSE*put.deltaK/put.index/put.index
        sigma = (sum(putComponent))*np.exp(T*R)*2/T
        sigma = sigma - (FF/FF_idx - 1)**2/T
    else:
        FF_idx = put.index[-1]
        try:
            if len(putAll.loc[FF_idx].CLOSE.values) > 1:
                put['CLOSE'].iloc[-1] = (putAll.loc[FF_idx].CLOSE.values[1] + callAll.loc[FF_idx].CLOSE.values[0])/2.0
        except:
            put['CLOSE'].iloc[-1] = (putAll.loc[FF_idx].CLOSE + callAll.loc[FF_idx].CLOSE)/2.0

        callComponent = call.CLOSE*call.deltaK/call.index/call.index
        putComponent  = put.CLOSE*put.deltaK/put.index/put.index
        sigma = (sum(callComponent)+sum(putComponent))*np.exp(T*R)*2/T
        sigma = sigma - (FF/FF_idx - 1)**2/T
    return sigma


def calGeneralSigmaSquare(options, FF, R, T):
    # 计算某个到期日期权对于GVIX的贡献sigma；
    # 输入为期权数据options，FF为forward index price，
    # R为无风险利率， T为期权剩余到期时间
    """
    params: options:该date为交易日的所有期权合约的基本信息和价格信息
            FF: 根据上一步计算得来的strike，然后再计算得到的forward index price， 根据它对所需要的看涨看跌合约进行划分。
                取小于FF的第一个行权价为中间行权价K0， 然后选取大于等于K0的所有看涨合约， 选取小于等于K0的所有看跌合约。
                对行权价为K0的看涨看跌合约，删除看涨合约，不过看跌合约的价格为两者的均值。
            R： 这部分期权合约到期日对应的无风险利率 shibor
            T： 还有多久到期（年化）
    return：Sigma：得到的结果是传入该到期日数据的Sigma
    """
    callAll = options[options.EXE_MODE == u"认购"].set_index(u"EXE_PRICE").sort_index()
    putAll = options[options.EXE_MODE == u"认沽"].set_index(u"EXE_PRICE").sort_index()
    callAll['deltaK'] = 0.05
    putAll['deltaK'] = 0.05

    # Interval between strike prices
    index = callAll.index
    if len(index) < 3:
        callAll['deltaK'] = index[-1] - index[0]
    else:
        for i in range(1, len(index) - 1):
            callAll['deltaK'].loc[index[i]] = (index[i + 1] - index[i - 1]) / 2.0
        callAll['deltaK'].loc[index[0]] = index[1] - index[0]
        callAll['deltaK'].loc[index[-1]] = index[-1] - index[-2]
    index = putAll.index
    if len(index) < 3:
        putAll['deltaK'] = index[-1] - index[0]
    else:
        for i in range(1, len(index) - 1):
            putAll['deltaK'].loc[index[i]] = (index[i + 1] - index[i - 1]) / 2.0
        putAll['deltaK'].loc[index[0]] = index[1] - index[0]
        putAll['deltaK'].loc[index[-1]] = index[-1] - index[-2]

    call = callAll[callAll.index > FF]
    put = putAll[putAll.index < FF]
    FF_idx = FF
    if put.empty:
        FF_idx = call.index[0]
        callComponent = call.CLOSE * call.deltaK / call.index / call.index

        callComponentGeneral = call.CLOSE * call.deltaK * (1+np.log(FF) - R*T - np.log(call.index)) / call.index / call.index
        sigmaGeneral = 2*np.exp(T * R)*sum(callComponentGeneral)

        sigma = (sum(callComponent)) * np.exp(T * R) * 2 / T
        sigma = sigma - (FF / FF_idx - 1) ** 2 / T
    elif call.empty:
        FF_idx = put.index[-1]
        putComponent = put.CLOSE * put.deltaK / put.index / put.index

        putComponentGeneral = put.CLOSE * put.deltaK * (1+np.log(FF) - R*T - np.log(put.index)) / put.index / put.index
        sigmaGeneral = 2*np.exp(T * R)*sum(putComponentGeneral)

        sigma = (sum(putComponent)) * np.exp(T * R) * 2 / T
        sigma = sigma - (FF / FF_idx - 1) ** 2 / T
    else:
        FF_idx = put.index[-1]
        try:
            if len(putAll.loc[FF_idx].CLOSE.values) > 1:
                put['CLOSE'].iloc[-1] = (putAll.loc[FF_idx].CLOSE.values[1] + callAll.loc[FF_idx].CLOSE.values[0]) / 2.0
        except:
            put['CLOSE'].iloc[-1] = (putAll.loc[FF_idx].CLOSE + callAll.loc[FF_idx].CLOSE) / 2.0

        callComponent = call.CLOSE * call.deltaK / call.index / call.index
        putComponent = put.CLOSE * put.deltaK / put.index / put.index

        callComponentGeneral = call.CLOSE * call.deltaK * (1+np.log(FF) - R*T - np.log(call.index)) / call.index / call.index
        putComponentGeneral = put.CLOSE * put.deltaK * (1+np.log(FF) - R*T - np.log(put.index)) / put.index / put.index
        sigmaGeneral = 2*np.exp(T * R)*(sum(callComponentGeneral) + sum(putComponentGeneral))

        sigma = (sum(callComponent) + sum(putComponent)) * np.exp(T * R) * 2 / T
        sigma = sigma - (FF / FF_idx - 1) ** 2 / T

    mu_T = -1*(sigma*T/2 - R*T)# 收益率的期望，由vix反推出来
    ln_K0_S0 = R*T - np.log(FF / FF_idx)# 期货定价公式的变形
    vega_T = ln_K0_S0**2 + 2 * ln_K0_S0 * (FF / FF_idx - 1) + sigmaGeneral# 收益率二阶矩，平方的期望
    gvixSigma = (vega_T - mu_T**2)/T
    return gvixSigma

def changeste(t):
    t = pd.Timestamp(t)
    str_t = t.strftime('%Y/%m/%d ')+'00:00'
    return str_t

def calDayVIX(vixDate,options_data,shibor_rate):
    # 利用CBOE的计算方法，计算历史某一日的未来30日期权波动率指数VIX
    """
    params：vixDate：计算VIX的日期  '%Y/%m/%d' 字符串格式
    return：VIX结果
    """
    if not isTradingDate(vixDate):
        return np.nan,np.nan
    # 拿取所需期权信息
    options = getHistDayOptions(vixDate,options_data)
    near, nexts = getNearNextOptExpDate(options, vixDate)
    shibor = periodsSplineRiskFreeInterestRate(options, vixDate,shibor_rate)
    R_near = shibor[datetime(near.year,near.month,near.day)]
    R_next = shibor[datetime(nexts.year,nexts.month,nexts.day)]

    if near == nexts:
        return np.nan,np.nan
    
    str_near = changeste(near)
    str_nexts = changeste(nexts)
    optionsNearTerm = options[options.EXE_ENDDATE == str_near]
    optionsNextTerm = options[options.EXE_ENDDATE == str_nexts]
    # time to expiration
    vixDate = datetime.strptime(vixDate,'%Y/%m/%d')
    T_near = (near - vixDate).days/365.0
    T_next = (nexts- vixDate).days/365.0
    # the forward index prices
    nearPriceDiff = getStrikeMinCallMinusPutClosePrice(optionsNearTerm)
    nextPriceDiff = getStrikeMinCallMinusPutClosePrice(optionsNextTerm)
    near_F = nearPriceDiff[0] + np.exp(T_near*R_near)*nearPriceDiff[1]
    next_F = nextPriceDiff[0] + np.exp(T_next*R_next)*nextPriceDiff[1]
    # 计算不同到期日期权对于VIX的贡献
    near_sigma = calSigmaSquare(optionsNearTerm, near_F, R_near, T_near)
    next_sigma = calSigmaSquare(optionsNextTerm, next_F, R_next, T_next)
    near_generalSigma = calGeneralSigmaSquare(optionsNearTerm, near_F, R_near, T_near)
    next_generalSigma = calGeneralSigmaSquare(optionsNextTerm, next_F, R_next, T_next)

    # 利用两个不同到期日的期权对VIX的贡献sig1和sig2，
    # 已经相应的期权剩余到期时间T1和T2；
    # 差值得到并返回VIX指数(%)
    # GVIX 同理
    w = (T_next - 30.0/365.0)/(T_next - T_near)
    vix = T_near*w*near_sigma + T_next*(1 - w)*next_sigma
    vix = 100*np.sqrt(abs(vix)*365.0/30.0)

    gvix = T_near*w*near_generalSigma + T_next*(1 - w)*next_generalSigma
    gvix = 100*np.sqrt(abs(gvix)*365.0/30.0)
    return vix,gvix


def calGVSpread():
    path = sys.path[0]

    shibor_rate = pd.read_csv(path + os.sep + 'shibor.csv', index_col=0, encoding='GBK')
    options_data = pd.read_csv(path + os.sep + 'options.csv', index_col=0)
    tradeday = pd.read_csv(path + os.sep + 'tradeday.csv')
    true_ivix = pd.read_csv(path + os.sep + 'ivixx.csv')


    ivix = []
    givix = []
    for day in tradeday['DateTime']:
        dayResult = calDayVIX(day,options_data,shibor_rate)
        ivix.append(dayResult[0])
        givix.append(dayResult[1])
        #print ivix

    from pyecharts import Line
    attr = true_ivix[u'日期'].tolist()
    line = Line(u"中国波指")
    line.add("中证指数发布", attr, true_ivix[u'收盘价(元)'].tolist(), mark_point=["max"])
    line.add("手动计算", tradeday['DateTime'].to_list(), ivix, mark_line=["max",'average'])
    line.render('vix.html')

    # 输出df
    result = pd.DataFrame([ivix,givix],index=['IVIX','GIVIX'],columns=tradeday['DateTime'].to_list()).T
    result['GV-Spread'] = result['GIVIX'] - result['IVIX']
    result.to_excel(path+os.sep+'IVIX.xlsx')

def updateGVSpread():
    path = sys.path[0]

    shiborRate = pd.read_csv(path+os.sep+"shibor.csv",index_col=0,encoding = "GBK")
    optionsData = pd.read_csv(path+os.sep+"options.csv",index_col=0)
    tradeDay = pd.read_csv(path+os.sep+"tradeday.csv")
    data = pd.read_excel(path+os.sep+"IVIX.csv",index_col=0)

    ivix = []
    givix = []
    dateList = [i for i in tradeDay['DateTime'] if i not in data.index]
    for day in dateList:
        dayResult = calDayVIX(day,optionsData,shiborRate)
        ivix.append(dayResult[0])
        givix.append(dayResult[1])

    # Output
    result = pd.DataFrame([ivix,givix], index=['IVIX','GIVIX'], columns=dateList).T
    result['GV-Spread'] = result['GIVIX'] - result['IVIX']
    res = pd.concat([data,result])
    from pyecharts import Line
    line = Line(u"China VIX")
    line.add("Re-Simulated VIX", res.index, res['GV_Spread'], mark_line=['max','average'])
    line.render("vix.html")
    res.to_excel(path+os.sep+"IVIX.xlsx")

    # Output Signal Png
    from RiskQuantLib.Tool.plotTool import plotLine
    plot = res[['GV-Spread']].iloc[-700:]
    sigma = plot['GV-Spread'].rolling(window=125).apply(np.nanstd)
    average = plot['GV-Spread'].rolling(window=40).apply(np.nanmean)
    plot['Upper 2 Sigma'] = average + 2 * sigma
    plot['Upper 1 Sigma'] = average + 1 * sigma
    plot['Mean'] = average
    plot['Lower 1 Sigma'] = average - 1 * sigma
    plot['Lower 2 Sigma'] = average - 2 * sigma

    plotLine(plot.iloc[-500:],"GV-Spread", "Time", "GV-Spread", path+os.sep+'GV-Spread.png')








