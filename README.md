# Stocks-Bot
A Python program that provides analysis of stocks to attempt to exploit their daily volatility to make large, short term returns.

After the Gamestop stock short debacle, I became interested in the stock market; particularly in how the volatility of some stocks meant one could double their money daily if they were able to predict when a sharp change in a stock might occur, and buy and sell at the right times. I wrote a Python program that used a parameterised statistical model to estimate when a stock was at its lowest or highest point, and indicate to buy or sell at that point. I trained it on past stock data obtained through the Yahoo Finance API, and after getting promising results I modified it to run on real-time data, and gave it the ability to "buy" and "sell" fake stocks and let it run over a two week period with a £1,000 initial investment. Unfortunately the Texas power crisis occurred in that two week period and the bot took crushing losses on the many energy stocks it was holding. I lost faith that the stock market was a reliable way to make income after that.

As I thus found out, it is practically impossible to guess whether, or when, stocks will change in value due to the efficient market hypothesis, and due to the heterscedastic nature of the stock market, it is reckless to try, so this project has remained a demonstration rather than a useful financial program.

It is currently capable of either running in real time, or by randomly picking past stock data to test its logic against. I intend to implement this as a sort of "Stock simulator" game at some point so wannabe day-traders can test their mettle against real-life scenarios.

