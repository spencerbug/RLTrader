import pytest
import random
import numpy as np
import collections
import time
import torch
import scipy.stats as stats
from analytics_server.analytics import fixeddepthchart_gen, ta_gen
from itertools import tee
from numpy_ringbuffer import RingBuffer
from numpy import testing as nptest
from numpy import inf
import matplotlib.pyplot as plt

def timestamp_generator(t0=None, n_step=520, dt=86400):
    """
    timestamp generator

    Arguments:
        t0: initial time. default now(), float seconds
        n_step: total steps in generator
        dt: change in time (float seconds) per step
    """
    if t0 is None:
        t0=time.time()
    for i in range(n_step):
        yield t0 + (i * dt)

def brownian_stock_generator(s0=100, mu=0.22, sigma=0.68, n_step=520, scaler=0.1):
    """
    Models a stock price S(t) using the Weiner process W(t) as
    `S(t) = S(0).exp{(mu-(sigma^2/2).t)+sigma.W(t)}`
    
    Arguments:
        s0: Iniital stock price, default 100
        mu: 'Drift' of the stock (upwards or downwards), default 0.2
        sigma: 'Volatility' of the stock, default 0.68
        n_step: Total number of datapoints generated
        scaler: multiplier for adjusting the period-to-period variance
    
    Returns:
        Generator with simulated stock prices over time-period deltaT
    """
    x0=0
    x1=x0

    # Weiner process with random normal distribution
    for i in range(0, n_step):
        x1 = sigma * (x0 + (np.random.normal() / np.sqrt(n_step)))
        stock_var = i * scaler * (mu-(sigma**2/2))
        yield s0 * np.exp(stock_var + x0)
        x0 = x1


def round_nearest(x, interval):
    return np.round(x/interval)*interval


# pg1, pg2, tsg need to be duplicates with itertools.tee()
def L2DataGenerator(pg1, pg2, vg, tsg, symbol='BTCUSD', dt=86400, stepsize=0.01):
    """ L2DataGenerator:
        Models Level 2 market data with a generator function
    Arguments:
        pg1: brownian generator for upper price
        pg2: brownian generator for lower price
        vg: brownian generator for volume
        tsg: timestamp generator (float seconds) 
        symbol: ticker symbol
        dt: minimum delta in time between outputs (float seconds)
        stepsize: minimum interval of prices
    Yields: object
    {
        "timestamp": timestamp of start of interval (float epoch),
        "endTimeStamp": time stamp of end of interval (float epoch)
        "symbol":symbol,
        "askPrices":[askprices...],
        "askVolumes":[askvolumes...],
        "bidPrices":[bidprices...],
        "bidVolumes":[bidvolumes...],
    }
    """

    for p1, p2, v, ts in zip(pg1, pg2, vg, tsg):
        hi=max(p1,p2)
        lo=min(p1,p2)
        mu=(hi+lo)/2
        sigma = np.random.uniform(0, 0.2*(hi-lo))
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
        a, b = (lo - mu) / sigma, (hi - mu) / sigma
        price_spread=round_nearest(stats.truncnorm(a, b, loc=mu, scale=sigma).rvs(int(v)), stepsize)
        v2p = collections.Counter(price_spread)
        vp=[(p, v2p[p]) for p in sorted(v2p.keys())]
        prices, volumes = zip(*vp)
        prices=np.array(prices)
        volumes=np.array(volumes)

        # currently volumes are in a gaussian curve shape,
        # reverse two halves to turn it into a gaussian bathtub curve
        volumes[:len(volumes)//2] = volumes[:len(volumes)//2][::-1]
        volumes[len(volumes)//2:] = volumes[len(volumes)//2:][::-1]

        bidprices=prices[:len(prices)//2]
        bidvolumes=volumes[:len(volumes)//2]
        askprices=prices[len(prices)//2:]
        askvolumes=volumes[len(volumes)//2:]

        yield {
            "timestamp": ts,
            "endTimestamp": ts + dt,
            "symbol":symbol,
            "askPrices":askprices,
            "askVolumes":askvolumes,
            "bidPrices":bidprices,
            "bidVolumes":bidvolumes,
        }

def OHLCGenerator(pg1, pg2, vg, tsg, symbol='BTCUSD'):
    """ OHLCGenerator:
        Generates fake open, high, low, close, volume
    Arguments:
        pg1: Price generator 1
        pg2: Price generator 2
        vg: volume generator
        tsg: timestamp generator
    """
    for p1, p2, v, ts in zip(pg1, pg2, vg, tsg):
        lo=min(p1,p2)
        hi=max(p1,p2)
        mu=(hi+lo)/2
        sigma = np.random.uniform(0, 0.2*(hi-lo))
        op, cl = np.random.normal(loc=mu, scale=sigma, size=2)
        yield {
            "timestamp": ts,
            "open": op,
            "high": hi,
            "low": lo,
            "close": cl,
            "volume": v
        }



def test_fixeddepthchart_gen_odd_midpoint():
    bucketindices=[1,2,4,7,10, inf]
    stepsize=1.0
    orderbook = {
        "bidPrices":  [1,4,5,6,7,11],
        "bidVolumes": [2,1,1,3,1,2],
        "askPrices": [12,13,16,17,30],
        "askVolumes": [1,2,4,3,1],
    }
    
    #bucketindices intervals, difference between the buckets
    #1,1,2,3,3,inf
    #bidprices: 11-1=10, 10-1=9, 9-2=7, 7-3=4, 4-3=1, 1-inf>0=0
    #askprices: 12+1=13, 13+1=14 14+2=16 16+3=19 19+3=22

    #pricesbucket are [[a1,b1],[a2,b2],[a3,b3],...] 
    # where bn=an+1
    # for bidprices, the volume v at price p fits in bucket at bucketindex n if an<p<=bn
    # for askprices, the volume v at price p fits in bucket at bucketindex n if an<=p<bn
    # the difference in equalities, is because the innermost price (closest to spread) should be inclusive, and outermost price should be exclusive, 
    # which taken to the extreme the farthest outermost price would be infinity for asks, and 0 for bids.
    # for bidprices the lowest price is outermost, and highest is innermost
    # for askprices the highest price is outermost, and the lowest is innermost


    expectedBidPrices=[[0,1],[1,4],[4,7],[7,9],[9,10],[10,11]]
    expectedAskPrices=[[12,13],[13,14],[14,16],[16,19],[19,22],[22,inf]]
    expectedBidVolumes=[2,1,5,0,0,2]
    expectedAskVolumes=[1,2,0,7,0,1]

    fdgc = fixeddepthchart_gen([orderbook], stepsize, bucketindices)
    for output in fdgc:
        nptest.assert_array_almost_equal(output["bidVolumes"], expectedBidVolumes)
        nptest.assert_array_almost_equal(output["askVolumes"], expectedAskVolumes)
        nptest.assert_array_almost_equal(output["bidPrices"], expectedBidPrices)
        nptest.assert_array_almost_equal(output["askPrices"], expectedAskPrices)

def test_fixeddepthchart_bugcase_010923():
    depthchart_buckets=[1,2,3,4,5,6,7,8,10,13,17,21,27,34,44,55,72,89,100]
    stepsize=0.1
    L2datapoint={
        "timestamp":1577844000.0,
        "endTimestamp":1577847600.0,
        "symbol":"BTCUSD",
        "askPrices": np.array([ 99.8,  99.9, 100. , 100.1, 100.2, 100.3, 100.4]),
        "askVolumes": np.array([  1,   3,  12,  19,  71, 121, 171]),
        "bidPrices": np.array([99.1, 99.2, 99.3, 99.4, 99.5, 99.6, 99.7]),
        "bidVolumes": np.array([214, 168, 123,  61,  29,  11,   2]),
    }

    fdcg = fixeddepthchart_gen([L2datapoint], stepsize, depthchart_buckets, False)
    fd=next(fdcg)
    nptest.assert_almost_equal(fd["bidPrices"], np.array([
       [89.7, 90.8],
       [90.8, 92.5],
       [92.5, 94.2],
       [94.2, 95.3],
       [95.3, 96.3],
       [96.3, 97. ],
       [97. , 97.6],
       [97.6, 98. ],
       [98. , 98.4],
       [98.4, 98.7],
       [98.7, 98.9],
       [98.9, 99. ],
       [99. , 99.1],
       [99.1, 99.2],
       [99.2, 99.3],
       [99.3, 99.4],
       [99.4, 99.5],
       [99.5, 99.6],
       [99.6, 99.7]]))



def test_pipeline():
    n_step=200
    t0=1577836800.000 #jan 1 2020 00:00
    dt=3600
    symbol='BTCUSD'
    stepsize=0.1
    depthchart_buckets=[1,2,3,4,5,6,7,8,10,13,17,21,27,34,44,55,72,89,100]
    pg1 = brownian_stock_generator(s0=95, n_step=n_step)
    pg2 = brownian_stock_generator(s0=105, n_step=n_step)
    vg = brownian_stock_generator(s0=1000, n_step=n_step)
    tsg = timestamp_generator(t0, n_step, dt)

    pg1_1, pg1_2 = tee(pg1, 2)
    pg2_1, pg2_2 = tee(pg2, 2)
    vg_1, vg_2 = tee(vg, 2)
    tsg_1, tsg_2 = tee(tsg, 2)

    L2g = L2DataGenerator(pg1_1, pg2_1, vg_1, tsg_1, symbol, dt, stepsize)
    ohlcg1, ohlcg2 = tee(OHLCGenerator(pg1_2, pg2_2, vg_2, tsg_2))

    fdcg = fixeddepthchart_gen(L2g, stepsize, depthchart_buckets, False)
    tag = ta_gen(ohlcg1, 100)
    datawidth=15 + (2*len(depthchart_buckets))
    rb = RingBuffer(100, dtype=(float, datawidth))

    fig=plt.figure(0)
    ax=fig.add_subplot(1,1,1)
    plt.firsttime=True

    for fd, ta , ohlc in zip(fdcg, tag, ohlcg2):
        prices=np.append(np.fromiter(reversed(fd["bidPrices"][:,0]), dtype=float), fd["askPrices"][:,0])
        volumes=np.append(np.fromiter(reversed(fd["bidVolumes"]), dtype=float), fd["askVolumes"])
        a=[
            ohlc["open"],
            ohlc["high"],
            ohlc["low"],
            ohlc["close"],
            ohlc["volume"],
            ta["macd"],
            ta["macd_sig"],
            ta["macd_div"],
            ta["adx"],
            ta["sma"],
            ta["rsi"],
            ta["obv"],
            ta["wvap"],
            ta["cci"],
            ta["tsi"],
            *volumes
        ]
        rb.append(a)

        if plt.firsttime:
            step,=ax.step(prices,volumes)
            plt.firsttime=False
            time.sleep(0.1)
        else:
            step.set_ydata(volumes)
            step.set_xdata(prices)
            plt.xlim([min(prices), max(prices)])
            plt.ylim([min(volumes), max(volumes)])
            plt.draw()
            time.sleep(0.1)
            plt.pause(0.0001)

    # if we got this far, the pipeline ran for a while and is working
    assert True == True


    # ob = {
    #     "ticket": 0,
    #     "gatewayResultCode": 0,
    #     "coreResultCode": 100,
    #     "description": "SUCCESS",
    #     "data": {
    #         "symbol": "BTCUSD",
    #         "askPrices": [
    #             2003
    #             2002.5
    #             2002
    #             2001
    #         ],
    #         "askVolumes": [
    #             103
    #             120
    #             200
    #             705
    #         ],
    #         "bidPrices": [

    #         ],
    #         "bidVolumes": [
                
    #         ]
    #     }
    # }