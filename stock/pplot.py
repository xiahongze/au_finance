import matplotlib.pylab as plt
import matplotlib.dates as mdates
from matplotlib.finance import candlestick_ohlc
import numpy as np


def moving_average(x, n, kind='simple'):
    """
    compute an n period moving average.
    kind is 'simple' | 'exponential' or 'exp'
    from: http://matplotlib.org/examples/pylab_examples/finance_work2.html
    """
    x = np.asarray(x)
    if kind == 'simple':
        weights = np.ones(n)
    elif kind == 'exp' or kind == 'exponential':
        weights = np.exp(np.linspace(-1., 0., n))
    else:
        raise ValueError("No such kind!")

    weights /= weights.sum()

    a = np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = a[n]
    return a


def plot_ohlc(df, maDay=5, maType='simple', **kwarg):
    """
    df: pandas DataFrame
        generated from yahoo_finance or it need to have these
        five columns:
        'Open', 'High', 'Low', 'Close', 'Volume'
    maDay: int
        number of days to do moving average
    maType: string
        'simple' or "exp"
    """
    # set default and unpack kwarg
    opt = {
        "title" : "Historical data",
        "xlabel" : "",
        "ylabel" : "Price",
        "lowerVolume" : 0,
        'colorup' : 'r',
        'colordown' : 'g'
    }
    opt.update(kwarg)
    
    # filter days when the market is not open.
    df = df[df['Volume']>opt['lowerVolume']].copy()
    
    # initialise figures
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(8,8))
    # adjust plot sizes
    l1, b1, w1, h1 = 0.10, 0.30, 0.85, 0.60 # top plot
    l2, b2, w2, h2 = 0.10, 0.10, 0.85, 0.20 # bottom plot
    ax1.set_position([l1, b1, w1, h1])
    ax2.set_position([l2, b2, w2, h2])

    # convert to mdates and plot volumes
    df['mdates'] = map(lambda date: mdates.date2num(date), df.index.to_pydatetime())
    df.plot(x='mdates', y='Volume', ax=ax2, legend=False, ls='steps')
    ax2.set_yscale("log")
    ax2.set_ylabel("Volume")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%h\n%Y'))
    
    # plot candlesticks
    sticks = candlestick_ohlc(
               ax1, df[['mdates', 'Open', 'High', 'Low', 'Close']].values,
               colorup=opt['colorup'], colordown=opt['colordown'],
               width=0.8, alpha=0.7)
    
    # create medium price
    df['median'] = df[['Open', 'High', 'Low', 'Close']].median(axis=1)
    df.plot(x='mdates', y='median', ax=ax1, kind='scatter', c='k', marker='_')
    # moving average
    maLabel = "{:d}D Moving Average".format(maDay)
    df['MA'] = moving_average(df['median'], maDay, maType) # true MA
    df[maLabel] = df['MA']+df['median'].mean()*0.05
    df.plot(x='mdates', y=maLabel, ax=ax1, c='m')
    
    # set title and other stuff
    ax1.set_title(opt["title"])
    ax1.set_ylabel(opt["ylabel"])
    ax2.set_xlabel(opt["xlabel"])
    plt.show()

if __name__ == "__main__":
    # import yahoo_finance as yf
    # df = yf.get_histdata('TOF')

    import pandas as pd
    df = pd.read_pickle("tof.pckl")
    df.to_pickle("tof.pckl")
    plot_ohlc(df,maType='exp',title='Historical data for TOF', lowerVolume=400)
    print df[(df['Volume']>1) & (df['Volume']<5000)]