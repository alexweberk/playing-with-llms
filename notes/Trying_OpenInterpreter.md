# OpenInterpreter を試す

今週流行っている OpenInterpreter を試してみました。

Github はこちらです：
https://github.com/KillianLucas/open-interpreter/

## ローカル環境で試す

```
pip install --upgrade open-interpreter
```

実行してみる。

```
interpreter
```

```
▌ Model set to GPT-4

Tip: To run locally, use interpreter --local

Open Interpreter will require approval before running code. Use interpreter -y to bypass this.

Press CTRL-C to exit.
```

コード実行に関して毎回自動で実行して OK の場合は `interpreter -y` で走らせると自動で実行してくれるようです。ちょっと怖いので一旦 `-y` はつけずに実行してみます。
質問をしてみる。

```
> 日本で大きな産業は何ですか？
```

```
日本の大きな産業には以下のようなものがあります：

   1 自動車産業：トヨタ、ホンダ、日産など、世界的に有名な自動車メーカーが多数存在します。
   2 電子機器産業：ソニー、パナソニック、シャープなど、高品質な電子機器を製造しています。
   3 鉄鋼産業：日本製鉄、JFEスチールなど、世界的に有名な鉄鋼メーカーがあります。
   4 化学産業：旭化成、住友化学など、化学製品の製造においても世界的に有名です。
   5 IT産業：ソフトバンク、ラクテンなど、IT関連の企業も多く存在します。

  これらは一部の例であり、他にも多くの産業が日本で活動しています。

>

  何か他にお手伝いできることがありますか？

```

明示的に OpenAI の API キーを設定していなかったのに解答がきた。

### Github にある例をコンソールで試してみる。

```

Plot APPL and META's normalized stock prices

```

(Python で実行する場合は `interpreter.chat("Plot APPL and META's normalized stock prices") `)

```
 To plot the normalized stock prices of APPL (Apple Inc.) and META (Meta Platforms Inc., formerly Facebook Inc.), we will need to follow these steps:

   1 Install necessary packages: We will need pandas, yfinance, and matplotlib. Pandas is a data manipulation library, yfinance is used to fetch historical market data from Yahoo finance, and matplotlib is used for creating static, animated, and interactive
     visualizations in Python.
   2 Fetch the stock data: We will fetch the historical stock data for APPL and META using yfinance.
   3 Normalize the stock prices: We will normalize the stock prices to compare them on the same scale.
   4 Plot the normalized stock prices: We will plot the normalized stock prices using matplotlib.

  Let's start with the first step, which is installing the necessary packages.



  pip install pandas yfinance matplotlib


  Would you like to run this code? (y/n)
```

実行するために `y` を入力。

```
The necessary packages have been installed successfully. Now, let's move on to the next step which is fetching the historical stock data for APPL (Apple Inc.) and META (Meta Platforms Inc.) using yfinance. We will fetch the data for the last 5 years.



  import yfinance as yf
  import pandas as pd

  # Define the ticker symbols
  tickers = ['AAPL', 'META']

  # Fetch the data
  data = yf.download(tickers, start='2017-01-01', end='2022-12-31')

  # Keep only the 'Adj Close' column
  data = data['Adj Close']

  # Display the first few rows of the dataframe
  data.head()


  Would you like to run this code? (y/n)
```

もう一度 `y` を入力。

```
import yfinance as yf

  import yfinance as yf
  import pandas as pd

  import yfinance as yf
  import pandas as pd


  import yfinance as yf
  import pandas as pd

  # Define the ticker symbols

  import yfinance as yf
  import pandas as pd

  # Define the ticker symbols

  import yfinance as yf
  import pandas as pd


  import yfinance as yf
  import pandas as pd

  # Define the ticker symbols
  tickers = ['AAPL', 'META']

  # Fetch the data
  data = yf.download(tickers, start='2017-01-01', end='2022-12-31')

  # Keep only the 'Adj Close' column
  data = data['Adj Close']

  # Display the first few rows of the dataframe
  data.head()


  [                       0%%                      ]
  [*********************100%%**********************]  2 of 2 completed
  AAPL        META
  Date
  2017-01-03  27.059311  116.860001
  2017-01-04  27.029024  118.690002
  2017-01-05  27.166477  120.669998
  2017-01-06  27.469330  123.410004
  2017-01-09  27.720940  124.900002


  We have successfully fetched the historical stock data for APPL (Apple Inc.) and META (Meta Platforms Inc.) for the last 5 years. The data includes the adjusted close prices for each trading day.

  Next, we will normalize the stock prices. Normalization is a scaling technique where the values are shifted and rescaled so that they end up ranging between 0 and 1. It is also known as Min-Max scaling. Here, we will normalize the stock prices so that the initial
  price for each stock is 1, and all subsequent prices are relative to the initial price. This will allow us to compare the performance of the two stocks on the same scale.



  # Normalize the stock prices
  data_normalized = data / data.iloc[0]

  # Display the first few rows of the normalized data
  data_normalized.head()


  Would you like to run this code? (y/n)
```

もう一度 `y` を入力。

```
# Normalize the stock prices
  data_normalized = data / data.iloc[0]

  # Display the first few rows of the normalized data
  data_normalized.head()


  AAPL      META
  Date
  2017-01-03  1.000000  1.000000
  2017-01-04  0.998881  1.015660
  2017-01-05  1.003960  1.032603
  2017-01-06  1.015153  1.056050
  2017-01-09  1.024451  1.068800


  The stock prices for APPL (Apple Inc.) and META (Meta Platforms Inc.) have been successfully normalized. The initial price for each stock is 1, and all subsequent prices are relative to the initial price.

  Finally, let's plot the normalized stock prices using matplotlib. This will allow us to visually compare the performance of the two stocks over the last 5 years.



  import matplotlib.pyplot as plt

  # Create a new figure
  plt.figure(figsize=(14, 7))

  # Plot the normalized stock prices
  for ticker in tickers:
      plt.plot(data_normalized[ticker], label=ticker)

  # Add a title and labels
  plt.title('Normalized Stock Prices (2017-2022)')
  plt.xlabel('Date')
  plt.ylabel('Normalized Price')
  plt.legend()

  # Display the plot
  plt.show()


  Would you like to run this code? (y/n)

```

もう一度 `y` を入力。

```
# Normalize the stock prices
data_normalized = data / data.iloc[0]

# Display the first few rows of the normalized data
data_normalized.head()


AAPL      META
Date
2017-01-03  1.000000  1.000000
2017-01-04  0.998881  1.015660
2017-01-05  1.003960  1.032603
2017-01-06  1.015153  1.056050
2017-01-09  1.024451  1.068800


The stock prices for APPL (Apple Inc.) and META (Meta Platforms Inc.) have been successfully normalized. The initial price for each stock is 1, and all subsequent prices are relative to the initial price.

Finally, let's plot the normalized stock prices using matplotlib. This will allow us to visually compare the performance of the two stocks over the last 5 years.



import matplotlib.pyplot as plt

# Create a new figure
plt.figure(figsize=(14, 7))

# Plot the normalized stock prices
for ticker in tickers:
    plt.plot(data_normalized[ticker], label=ticker)

# Add a title and labels
plt.title('Normalized Stock Prices (2017-2022)')
plt.xlabel('Date')
plt.ylabel('Normalized Price')
plt.legend()

# Display the plot
plt.show()


Would you like to run this code? (y/n)
```

もう一度 `y` を入力。

```

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# Create a new figure

import matplotlib.pyplot as plt

# Create a new figure

import matplotlib.pyplot as plt

# Create a new figure

plt.figure(figsize=(14, 7))

import matplotlib.pyplot as plt

# Create a new figure

plt.figure(figsize=(14, 7))

import matplotlib.pyplot as plt

# Create a new figure

plt.figure(figsize=(14, 7))

import matplotlib.pyplot as plt

# Create a new figure

plt.figure(figsize=(14, 7))

# Plot the normalized stock prices

for ticker in tickers:
plt.plot(data_normalized[ticker], label=ticker)

# Add a title and labels

plt.title('Normalized Stock Prices (2017-2022)')
plt.xlabel('Date')
plt.ylabel('Normalized Price')
plt.legend()

# Display the plot

plt.show()
█

  <Figure size 1400x700 with 0 Axes>
  [<matplotlib.lines.Line2D object at 0x1675058d0>]
  [<matplotlib.lines.Line2D object at 0x16be2d710>]
  Text(0.5, 1.0, 'Normalized Stock Prices (2017-2022)')
  Text(0.5, 0, 'Date')
  Text(0, 0.5, 'Normalized Price')
  <matplotlib.legend.Legend object at 0x16764b310>

```

無事グラフが生成されました！
<img />

閉じたあとに、

```

Here is the plot of the normalized stock prices for APPL (Apple Inc.) and META (Meta Platforms Inc.) from 2017 to 2022. As you can see, both stocks have shown significant growth over this period, but the growth has not been linear and there have been periods of
both increase and decrease.

Please note that this is a simplified analysis and does not take into account many factors that could influence the performance of these stocks. For a more comprehensive analysis, you may want to consider other factors such as earnings reports, economic indicators,
and market trends.

Is there anything else you would like to know or do?

```

## 終わりに

OpenInterpreter ですが、ChatGPT の Code Interpreter と似た感覚で使え非常に便利そうでした。
これだけ流行っているのもうなずけます。
Terminal 上で動くのでインプットファイルを使った編集などもかなり楽そうでした。

まだ試してませんが、Python で走らせることもできるみたいです。

```python

import interpreter

interpreter.chat("Plot APPL and META's normalized stock prices") # Executes a single command
interpreter.chat() # Starts an interactive chat

```

以上、お読みいただきありがとうございます。少しでも参考になればと思います。
もし今回の記事が楽しんでいただけたのであれば、note と Twitter でフォローしていただけると嬉しいです。

https://twitter.com/alexweberk

## 参考文献

- [OpenInterpreter Github Repo](https://github.com/KillianLucas/open-interpreter/)

#note #機械学習
#LLM
