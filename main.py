import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import praw
from psaw import PushshiftAPI
import datetime
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
import csv
import random
import time
import winsound
import stocks_lists as s

def getStockData(stock,prdString,intvlString):
    data = yf.download(tickers=stock,period=prdString,interval=intvlString)
    return(data)

def simplePlotStock(data):
    s = data['Open'].to_numpy()
    s = s[s > 0]
    x = np.linspace(0,len(s)-1,len(s))
    plt.plot(x,s,'r--')
    plt.show()

def getStockIntervalTimeTick(data):
    # returns number of ticks in each day, so one can isolate market days for graph analysis
    dates = data.index.tolist()
    dates = [str(i.date()) for i in dates]
    uniqueDates = []
    count = []
    dateOrder = []
    for i in range(len(dates)):
        if(dates[i] in uniqueDates):
            pass
            count[len(count)-1] += 1
        else:
            uniqueDates.append(dates[i])
            count.append(1)
    return(uniqueDates,count)

def normalisedDailyDistribution(dailyData,intvlData,uniqueDates,count,stat='Open'):
    sd = dailyData[stat].tolist()
    si = intvlData[stat].tolist()
    sd = [float(i) for i in sd]
    si = [float(i) for i in si]
    normalisedDays = {}
    for i in range(len(uniqueDates)):
        normalisedDays[uniqueDates[i]] = []
        totalCount = 0
        for j in range(i):
            totalCount += count[j]
        for j in range(count[i]):
            normalisedDays[uniqueDates[i]].append(si[totalCount+j]-sd[i])
    return(normalisedDays,sd)

def overlappingDictPlot(stockDict,sd):
    entries = [0]
    for key in stockDict:
        for i in range(len(stockDict[key])):
            if(len(entries) <= i):
                entries.append(1)
            else:
                entries[i] += 1
        plt.plot(np.linspace(0,len(stockDict[key])-1,len(stockDict[key])),stockDict[key],'r--')
    avgValue = [0 for i in entries]
    highAvg = [0 for i in entries]
    lowAvg = [0 for i in entries]
    maxDaily = max(sd)
    minDaily = min(sd)
    dayKey = 0
    for key in stockDict:
        for i in range(len(stockDict[key])):
            avgValue[i] += stockDict[key][i]/entries[i]
            highAvg[i] += (stockDict[key][i]/entries[i])*((sd[dayKey]-minDaily)/(maxDaily-minDaily))
            lowAvg[i] += (stockDict[key][i]/entries[i])*((maxDaily-sd[dayKey])/(maxDaily-minDaily))
        dayKey += 1
    plt.plot(np.linspace(0,len(avgValue)-1,len(avgValue)),avgValue,'b--')
    plt.plot(np.linspace(0,len(highAvg)-1,len(highAvg)),highAvg,'g--')
    plt.plot(np.linspace(0,len(lowAvg)-1,len(lowAvg)),lowAvg,'r--')
    plt.show()

def calcRollingAverage(data,avgInterval):
    averages = []
    for i in range(avgInterval):
        averages.append(sum(data[0:i+1])/(i+1))
    for i in range(avgInterval,len(data)):
        averages.append(sum(data[i-avgInterval:i])/avgInterval)
    return(averages)

def getStockDataForAnalysis(stocks,backdatePeriod="7d",pollInterval="1m",allTimeCheck="2y",cat='Open'):
    stockData = [[]]
    longStockData = [[]]
    recentDates = []
    for i in range(len(stocks)):
        data = getStockData(stocks[i],backdatePeriod,pollInterval)
        dates = data.index.tolist()
        if(len(dates) > 0):
            recentDates.append(dates[len(dates)-1])
        else:
            recentDates.append(0)
        longData = getStockData(stocks[i],allTimeCheck,"1d")
        if(i == 0):
            stockData[0] = data[cat].tolist()
            longStockData[0] = longData[cat].tolist()
        else:
            stockData.append(data[cat].tolist())
            longStockData.append(longData[cat].tolist())
    return(stockData,longStockData,recentDates,data)

def clipData(data,stockData,longStockData,clipDuration=7):
    # assumes longStockData is always at daily intervals, and only clips data in multiples of days
    uniqueDates,count = getStockIntervalTimeTick(data)
    preStockData = [[]]
    preLongData = [[]]
    postStockData = [[]]
    postLongData = [[]]
    count = count[len(count)-clipDuration:]
    clip = sum(count)
    
    for i in range(len(stockData)):
        if(i == 0):
            preStockData[i] = stockData[i][:len(stockData[i])-clip]
            postStockData[i] = stockData[i][len(stockData[i])-clip:]
            preLongData[i] = longStockData[i][:len(longStockData[i])-clipDuration]
            postLongData[i] = longStockData[i][len(longStockData[i])-clipDuration:]
        else:
            preStockData.append(stockData[i][:len(stockData[i])-clip])
            postStockData.append(stockData[i][len(stockData[i])-clip:])
            preLongData.append(longStockData[i][:len(longStockData[i])-clipDuration])
            postLongData.append(longStockData[i][len(longStockData[i])-clipDuration:])
    return(preStockData,postStockData,preLongData,postLongData,count)

def progressClippedData(preStockData,postStockData,preLongData,postLongData,count):
    for i in range(len(preStockData)):
        if(len(postStockData[i]) > 0):
            preStockData[i].pop(0)
            preStockData[i].append(postStockData[i][0])
            postStockData[i].pop(0)
            if(len(postStockData[i]) < sum(count)):
                count.pop(0)
                preLongData[i].pop(0)
                preLongData[i].append(postLongData[i][0])
                postLongData[i].pop(0)
    return(preStockData,postStockData,preLongData,postLongData,count)

def sortByHighest(values):
    sortedValues = []
    sortedRanks = []
    for j in range(len(values)):
        tempArray = [i for i in sortedValues]
        i = 0
        if(len(sortedValues) == 1):
            if(values[j] <= sortedValues[0]):
                i = 1
        while(len(tempArray) > 1):
            if(values[j] > tempArray[int(len(tempArray)/2)]):
                tempArray = tempArray[:int(len(tempArray)/2)]
            elif(values[j] < tempArray[int(len(tempArray)/2)]):
                i += int(len(tempArray)/2)
                tempArray = tempArray[int(len(tempArray)/2):]
            elif(values[j] == tempArray[int(len(tempArray)/2)]):
                i += int(len(tempArray)/2)
                break
            if(len(tempArray) == 1):
                if(values[j] <= tempArray[0]):
                    i += 1
        sortedValues.insert(i,values[j])
        sortedRanks.insert(i,j)
    return(sortedValues,sortedRanks)

def analyseDips(stocks,stockData,longStockData,avgInterval=15):
    # stocks is an array of the stock names (e.g. ["TSLA","GME"]
    # all time check is used to provide analysis on longer term movements (always in days!)
    # backdatePeriod is duration to store stock data over, that will be used to analyse the shape of stock curve
    # pollInterval is how often to grab data, avgInterval is time IN UNITS OF POLLINTERVAL to take rolling average of curve gradient over
    highs = []
    lows = []
    mediumHigh = []
    mediumLow = []
    longHigh = []
    longLow = []
    for i in range(len(stockData)):
        highs.append(max(stockData[i]))
        lows.append(min(stockData[i]))
        longHigh.append(max(longStockData[i]))
        longLow.append(min(longStockData[i]))
    for i in range(len(longLow)):
        if(longLow[i] == 0 or longHigh[i] == longLow[i]):
            print(i,stocks[i])
            print(longStockData[i])
    medians = [(highs[i]+lows[i])/2 for i in range(len(highs))]
    means = [sum(stockData[i])/len(stockData[i]) for i in range(len(stockData))]
    longMeans = [sum(longStockData[i])/len(longStockData[i]) for i in range(len(longStockData))]
    longMedians = [longLow[i]+(longHigh[i]-longLow[i])/2 for i in range(len(longLow))]
    largestRise = []
    largestDip = []
    for i in range(len(stockData)):
        rise = 0.0
        dip = 0.0
        for j in range(avgInterval,len(stockData[i])):
            if(stockData[i][j] - stockData[i][j-avgInterval] > rise):
                rise = stockData[i][j] - stockData[i][j-avgInterval]
            elif(stockData[i][j] - stockData[i][j-avgInterval] < dip):
                dip = stockData[i][j] - stockData[i][j-avgInterval]
        largestRise.append(rise)
        largestDip.append(dip)
    averages = [[]]
    percChanges = [[]]
    longPercChanges = [[]]
    scores = []
    for i in range(len(stockData)):
        average = calcRollingAverage(stockData[i],avgInterval) # currently not in use
        percChange = [100*(stockData[i][j+1]-stockData[i][j])/stockData[i][j] for j in range(len(stockData[i])-1)]
        longPercChange = [100*(longStockData[i][j+1]-longStockData[i][j])/longStockData[i][j] for j in range(len(longStockData[i])-1)]
        if(i == 0):
            averages[0] = average
            percChanges[0] = percChange
            longPercChanges[0] = longPercChange
        else:
            averages.append(average)
            percChanges.append(percChange)
            longPercChanges.append(longPercChange)
            
    for i in range(len(stockData)):
        percDifOverPeriod = sum(percChanges[i])
        percDifOverQuarter = sum(percChanges[i][int(3*len(percChanges[i])/4):])
        percDifLastInterval = sum(percChanges[i][len(percChanges[i])-1-avgInterval:])
        longPercDifPeriod = sum(longPercChanges[i])
        longQuarterPercDif = sum(longPercChanges[i][int(3*len(longPercChanges[i])/4):])
        
        percSensFactor = 1/(100*((highs[i]-lows[i])/lows[i]))
        longPercSensFactor = 1/(100*((longHigh[i]-longLow[i])/longLow[i]))
        meanFactor = (medians[i]-stockData[i][len(stockData[i])-1])/min(stockData[i][len(stockData[i])-1],medians[i])*100
        gradientFactor = (1+abs(percDifLastInterval*percSensFactor))*(1-abs(percDifOverQuarter*percSensFactor))
        durationFactor = (longMeans[i]/longMedians[i])*(1-longPercDifPeriod*longPercSensFactor) # helps to route out "squeezed" stock like GME and SPWR
        score = meanFactor*gradientFactor*durationFactor
        scores.append(score)
        
    sortedValues,sortedRanks = sortByHighest(scores)
    
    return(scores,sortedValues,sortedRanks)

def stockBotSim(stocks,portfolio,scoreTargetBuy=5,scoreTargetSell=5):
    stockData,longStockData,recentDates,data = getStockDataForAnalysis(stocks,"10d","5m","1y",'Close')
    delIndex = []
    for i in range(len(stockData)):
        if(len(stockData[i]) == 0):
            delIndex.append(i)
    deletions = 0
    for i in range(len(delIndex)):
        stockData.pop(delIndex[i]-deletions)
        longStockData.pop(delIndex[i]-deletions)
        stocks.pop(delIndex[i]-deletions)
        deletions += 1
        
    movements = []

    preStockData,postStockData,preLongData,postLongData,count = clipData(data,stockData,longStockData,5)
    totalScores = [[] for i in range(len(preStockData))]
    stockPrice = [[] for i in range(len(preStockData))]
    pastPrice = [[preStockData[i][j] for j in range(len(preStockData[i]))] for i in range(len(preStockData))]
    lowestEquity = portfolio[5]
    
    while(len(postStockData[0]) > 1):
        preStockData,postStockData,preLongData,postLongData,count = progressClippedData(preStockData,
                                                        postStockData,preLongData,postLongData,count)
        scores,sortedScores,sortedRanks = analyseDips(stocks,preStockData,preLongData,5)
        
        for j in range(len(scores)):
            totalScores[j].append(scores[j])
            stockPrice[j].append(preStockData[j][len(preStockData[j])-1])

            if(scores[j] > scoreTargetBuy):
                price = 50*int(scores[j]/scoreTargetBuy) # left in in case variable price for stocks you already hold
                if(stocks[j] in portfolio[0]):
                    index = portfolio[0].index(stocks[j])
                    if(portfolio[5] >= price and portfolio[3][index]/stockPrice[j][len(stockPrice[j])-1] > 1.02): # only buy stock if price is 2% lower than what you hold
                        newShares = price/stockPrice[j][len(stockPrice[j])-1]
                        portfolio[4][index] = (portfolio[4][index]*portfolio[1][index]+scores[j]*newShares)/(portfolio[1][index]+newShares)
                        portfolio[2][index] = (portfolio[2][index]*portfolio[1][index]+newShares*stockPrice[j][len(stockPrice[j])-1])/(portfolio[1][index]+newShares)
                        portfolio[1][index] += newShares
                        if(portfolio[3][index] > stockPrice[j][len(stockPrice[j])-1]):
                            portfolio[3][index] = stockPrice[j][len(stockPrice[j])-1]
                        portfolio[5] -= price
                        movements.append(["buy",newShares,stocks[j],stockPrice[j][len(stockPrice[j])-1],scores[j],len(postStockData[0])])
                elif(portfolio[5] >= price): 
                        portfolio[0].append(stocks[j])
                        portfolio[1].append(price/stockPrice[j][len(stockPrice[j])-1])
                        portfolio[2].append(stockPrice[j][len(stockPrice[j])-1])
                        portfolio[3].append(stockPrice[j][len(stockPrice[j])-1])
                        portfolio[4].append(scores[j])
                        portfolio[5] -= price
                        movements.append(["buy",portfolio[1][len(portfolio[1])-1],stocks[j],stockPrice[j][len(stockPrice[j])-1],
                                                                              scores[j],len(postStockData[0])])
            elif(stocks[j] in portfolio[0]):
                index = portfolio[0].index(stocks[j])
                scores[j] = scores[j] - (stockPrice[j][len(stockPrice[j])-1]/portfolio[2][index]-1)*100
                if(scores[j] < -scoreTargetSell):
                    if(stockPrice[j][len(stockPrice[j])-1]/portfolio[2][index] > 1.02 or scores[j] < -scoreTargetSell*4):
                        portfolio[5] += stockPrice[j][len(stockPrice[j])-1]*portfolio[1][index]
                        movements.append(["sell",portfolio[1][index],stocks[j],stockPrice[j][len(stockPrice[j])-1],
                                                          portfolio[2][index],portfolio[4][index],scores[j],len(postStockData[0])])
                        portfolio[0].pop(index)
                        portfolio[1].pop(index)
                        portfolio[2].pop(index)
                        portfolio[3].pop(index)
                        portfolio[4].pop(index)
                    else:
                        pass
                else:
                    pass
            if(portfolio[5] < lowestEquity):
                lowestEquity = portfolio[5]

    for i in range(len(movements)):
        print(movements[i])
    profits = []
    bidValue = 0
    for i in range(len(portfolio[0])):
        index = stocks.index(portfolio[0][i])
        profits.append((stockPrice[index][len(stockPrice[index])-1]/portfolio[2][i]-1)*100)
        bidValue += stockPrice[index][len(stockPrice[index])-1]*portfolio[1][i]
    print("% gain on current bid value of assets:")
    print(portfolio[0])
    print(profits)
    returns = bidValue+portfolio[5]
    print("Total bid value of portfolio (inc cash):",returns)
    '''
    for j in range(len(totalScores)):
        fig,axes = plt.subplots(3)
        fig.suptitle(stocks[j])
        axes[0].plot(np.linspace(0,len(totalScores[j])-1,len(totalScores[j])),totalScores[j],'r--')
        axes[1].plot(np.linspace(0,len(stockPrice[j])-1,len(stockPrice[j])),stockPrice[j],'b--')
        axes[2].plot(np.linspace(0,len(pastPrice[j])-1,len(pastPrice[j])),pastPrice[j],'g--')
        plt.show()
    '''
    return(returns,lowestEquity,portfolio)

def runSim(allStocks,numSim,portfolio,numRuns=100,buyScore=5,sellScore=5):
    overallReturns = []
    lowestEquities = []
    startPortfolio = [[portfolio[i][j] for j in range(len(portfolio[i]))] for i in range(len(portfolio)-1)]
    startPortfolio.append(portfolio[5])
    for sim in range(numRuns):
        stocks = [i for i in allStocks]
        for i in range(len(allStocks)-numSim):
            index = random.randrange(len(stocks))
            stocks.pop(index)
        print(stocks)
        returns,lowestEquity,portfolio = stockBotSim(stocks,portfolio,buyScore,sellScore)
        overallReturns.append(returns)
        lowestEquities.append(lowestEquity)
        portfolio = [[startPortfolio[i][j] for j in range(len(startPortfolio[i]))] for i in range(len(startPortfolio)-1)]
        portfolio.append(startPortfolio[5])
    for i in range(len(overallReturns)):
        print(overallReturns[i],lowestEquities[i])

def liveStockBot(stocks,freq=60,analysisPeriod="5d",interval="5m",backdatePeriod="300d",stat='Close'):
    startTime = time.time()
    hourlyReview = time.time()
    lastDate = "start date"
    movements = loadMovements()

    while(True):
        frameStart = time.time()
        stocks = list(dict.fromkeys(stocks))
        stockData,longStockData,dates,data = getStockDataForAnalysis(stocks,analysisPeriod,interval,backdatePeriod,stat)
        
        delIndex = []
        for i in range(len(stockData)):
            subDelIndex = []
            if(len(stockData[i]) == 0 or len(longStockData[i]) == 0):
                delIndex.append(i)
            elif(max(stockData[i]) == min(stockData[i]) or max(longStockData[i]) == min(longStockData[i])):
                delIndex.append(i)
            for j in range(len(stockData[i])):
                if(math.isnan(stockData[i][j]) or stockData[i][j] == 0):
                    subDelIndex.append(j)
            deletions = 0
            for j in range(len(subDelIndex)):
                stockData[i].pop(subDelIndex[j]-deletions)
                deletions += 1
            subDelIndex = []
            for j in range(len(longStockData[i])):
                if(math.isnan(longStockData[i][j]) or longStockData[i][j] == 0):
                    subDelIndex.append(j)
            deletions = 0
            for j in range(len(subDelIndex)):
                longStockData[i].pop(subDelIndex[j]-deletions)
                deletions += 1
        
        delIndex = list(dict.fromkeys(delIndex))                
        deletions = 0
        for i in range(len(delIndex)):
            stockData.pop(delIndex[i]-deletions)
            longStockData.pop(delIndex[i]-deletions)
            stocks.pop(delIndex[i]-deletions)
            dates.pop(delIndex[i]-deletions)
            deletions += 1

        currMovements = []
        scores,scoreValues,sortedRanks = analyseDips(stocks,stockData,longStockData,15)
        currDate = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        
        for j in range(len(scores)):
            scoreTargetBuy = 3
            scoreTargetSell = 3
            if(scores[j] > scoreTargetBuy):
                price = 50*int(scores[j]/scoreTargetBuy) # left in in case variable price for stocks you already hold
                if(stocks[j] in portfolio[0]):
                    index = portfolio[0].index(stocks[j])
                    if(portfolio[5] >= price and portfolio[3][index]/stockData[j][len(stockData[j])-1] > 1.02): # only buy stock if price is 2% lower than what you hold
                        newShares = price/stockData[j][len(stockData[j])-1]
                        portfolio[4][index] = (portfolio[4][index]*portfolio[1][index]+scores[j]*newShares)/(portfolio[1][index]+newShares)
                        portfolio[2][index] = (portfolio[2][index]*portfolio[1][index]+newShares*stockData[j][len(stockData[j])-1])/(portfolio[1][index]+newShares)
                        portfolio[1][index] += newShares
                        if(portfolio[3][index] > stockData[j][len(stockData[j])-1]):
                            portfolio[3][index] = stockData[j][len(stockData[j])-1]
                        portfolio[5] -= price
                        movements.append(["buy",newShares,stocks[j],stockData[j][len(stockData[j])-1],scores[j],currDate])
                        currMovements.append(["buy",newShares,stocks[j],stockData[j][len(stockData[j])-1],scores[j],currDate])
                elif(portfolio[5] >= price): 
                        portfolio[0].append(stocks[j])
                        portfolio[1].append(price/stockData[j][len(stockData[j])-1])
                        portfolio[2].append(stockData[j][len(stockData[j])-1])
                        portfolio[3].append(stockData[j][len(stockData[j])-1])
                        portfolio[4].append(scores[j])
                        portfolio[5] -= price
                        movements.append(["buy",portfolio[1][len(portfolio[1])-1],stocks[j],stockData[j][len(stockData[j])-1],
                                                                              scores[j],currDate])
                        currMovements.append(["buy",portfolio[1][len(portfolio[1])-1],stocks[j],stockData[j][len(stockData[j])-1],
                                                                              scores[j],currDate])
            if(stocks[j] in portfolio[0]):
                index = portfolio[0].index(stocks[j])
                scores[j] = scores[j] - (stockData[j][len(stockData[j])-1]/portfolio[2][index]-1)*100
                if(scores[j] < -scoreTargetSell):
                    #if(stockData[j][len(stockData[j])-1]/portfolio[2][index] > 1.02 or scores[j] < -scoreTargetSell*4):
                    portfolio[5] += stockData[j][len(stockData[j])-1]*portfolio[1][index]
                    movements.append(["sell",portfolio[1][index],stocks[j],stockData[j][len(stockData[j])-1],
                                                      portfolio[2][index],portfolio[4][index],scores[j],currDate])
                    currMovements.append(["sell",portfolio[1][index],stocks[j],stockData[j][len(stockData[j])-1],
                                                      portfolio[2][index],portfolio[4][index],scores[j],currDate])
                    portfolio[0].pop(index)
                    portfolio[1].pop(index)
                    portfolio[2].pop(index)
                    portfolio[3].pop(index)
                    portfolio[4].pop(index)
                    #else:
                    #    pass
                else:
                    pass
        print("-----------------------------------------------------")
        recentDate = 0
        activeStocks = []
        for i in range(len(dates)):
            if(dates[i].timestamp() > recentDate):
                recentDate = dates[i].timestamp()
        for i in range(len(dates)):
            if(dates[i].timestamp() > recentDate - 1800):
                activeStocks.append(stocks[i]+" (O)")
            else:
                activeStocks.append(stocks[i])
        recentDate = datetime.datetime.fromtimestamp(recentDate)
        print("Most recent data = ",str(recentDate.strftime("%m/%d/%Y, %H:%M:%S")))
    
        stockOrder = []
        dates = [dates[sortedRanks[i]] for i in range(len(dates))]
        for i in range(len(sortedRanks)):
            stockOrder.append(activeStocks[sortedRanks[i]])
            scoreValues[i] = round(scoreValues[i],2)
        print(stockOrder)
        print(scoreValues)
        if(len(currMovements) > 0):
            duration = 3000
            freq = 440
            winsound.Beep(freq,duration)
            print("RECOMMENDED MOVEMENTS:")
            for i in range(len(currMovements)):
                print(currMovements[i][0],currMovements[i][1],currMovements[i][2],"at",currMovements[i][3])
                if(currMovements[i][0] == "sell"):
                    print("Bought at",round(currMovements[i][4],1),"at score",round(currMovements[i][5],1),"(current score =",round(currMovements[i][6],1),")",
                          "timestamp:",currMovements[i][7])
                else:
                    print("Current score:",currMovements[i][4],"timestamp:",currMovements[i][5])
        savePortfolio(portfolio)
        saveMovements(movements)

        if(time.time() - hourlyReview >= 3600):
            hourlyReview = time.time()
            print("====================================================")
            print("HOURLY PORTFOLIO REVIEW:")
            print("Stock - units - Avg buy - Lowest buy - Score at buy - Current % profit     - cash = ",round(portfolio[5],2))
            bidValue = 0
            for i in range(len(portfolio[0])):
                index = stocks.index(portfolio[0][i])
                print(portfolio[0][i],round(portfolio[1][i],2),round(portfolio[2][i],2),round(portfolio[3][i],2),round(portfolio[4][i],2),
                      round((stockData[index][len(stockData[index])-1]/portfolio[2][i]-1)*100,2))
                bidValue += stockData[index][len(stockData[index])-1]*portfolio[1][i]
            returns = bidValue+portfolio[5]
            print("Current value of portfolio (inc cash):",returns)

        while(time.time() < frameStart + freq):
            time.sleep(10)

def savePortfolio(portfolio):
    # # stock, units, avg buy price, lowest buy price, score at buy, cash
    with open("portfolio.dat","w+") as portfolioFile:
        for i in range(len(portfolio)):
            portfolioFile.write(str(portfolio[i]).replace("[","").replace("]","").replace("'","").replace('"','').replace(" ","")+"\n")

def loadPortfolio():
    # stock, units, avg buy price, lowest buy price, score at buy, cash
    try:
        with open("portfolio.dat", newline='') as portfolioFile:
            portfolio = list(csv.reader(portfolioFile))
        for i in range(len(portfolio[0])):
            portfolio[1][i] = float(portfolio[1][i])
            portfolio[2][i] = float(portfolio[2][i])
            portfolio[3][i] = float(portfolio[3][i])
            portfolio[4][i] = float(portfolio[4][i])
        portfolio[5] = float(portfolio[5][0])
    except IndexError:
        portfolio = [[],[],[],[],[],1000]
    except FileNotFoundError:
        portfolio = [[],[],[],[],[],1000]
        f = open("portfolio.dat", "w+")
        for i in portfolio:
            f.write(str(i))
        f.close()
    return(portfolio)

def saveMovements(movements):
    with open("movements.dat","w+") as movementsFile:
        for i in range(len(movements)):
            movementsFile.write(str(movements[i]).replace("[","").replace("]","").replace("'","").replace('"','').replace(" ","")+"\n")

def loadMovements():
    try:
        with open("movements.dat", newline='') as movementsFile:
            movements = list(csv.reader(movementsFile))
    except:
        movements = []
    return(movements)

if(__name__=="__main__"):
    stocks = s.renewables_eToro
    portfolio = loadPortfolio()
    # Run one of the below functions - first runs in real time, bottom provides a simulation of the stocks guesswork using past stocks
    #liveStockBot(stocks,freq=900,analysisPeriod="5d",interval="1m",backdatePeriod="300d",stat='Close')
    runSim(stocks,len(stocks),portfolio,1,3,3)
    
