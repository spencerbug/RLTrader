import pytest
import random
import numpy as np
import collections
import time


def brownian_stock_generator(s0=100, mu=0.22, sigma=0.68, deltaT=52, dt=0.1):
    """
    Models a stock price S(t) using the Weiner process W(t) as
    `S(t) = S(0).exp{(mu-(sigma^2/2).t)+sigma.W(t)}`
    
    Arguments:
        s0: Iniital stock price, default 100
        mu: 'Drift' of the stock (upwards or downwards), default 0.2
        sigma: 'Volatility' of the stock, default 0.68
        deltaT: The time period for which the future prices are computed, default 52 (as in 52 weeks)
        dt (optional): The granularity of the time-period, default 0.1
    
    Returns:
        Generator with simulated stock prices over time-period deltaT
    """
    x0=0
    x1=x0
    n_step = int(deltaT/dt)
    t_step = deltaT / n_step

    # Weiner process with random normal distribution
    for i in range(0, n_step):
        x1 = sigma * (x0 + (np.random.normal() / np.sqrt(n_step)))
        stock_var = i * t_step * (mu-(sigma**2/2))
        yield s0 * np.exp(stock_var + x0)
        x0 = x1

def round_nearest(x, ticksize):
    return np.round(x/ticksize)*ticksize

def L2DataGenerator(symbol='BTCUSD', init_price=100, init_volume=1000, ticksize=0.01, mu=0.22, sigma=0.68, deltaT=52, dt=0.1):
    """
    Models Level 2 market data with a generator function
    arguments:
    symbol: ticker symbol
    init_volume: initial volume in market, # unit bids + # unit ask. drifts with mu, sigma
    ticksize: minimum interval of prices
    mu: 'Drift' of the stock (upwards or downwards), default 0.22
    sigma: 'Volatility' of the stock, default 0.68
    deltaT: The time period for which the future prices are computed, default 52 (as in 52 weeks)
    dt (optional): The granularity of the time-period, default 0.1
    """
    pricegen=brownian_stock_generator(init_price, mu, sigma, deltaT, dt)
    volumegen=brownian_stock_generator(init_volume, mu, sigma, deltaT, dt)
    for price, volume in zip(pricegen, volumegen):
        price_spread=round_nearest(np.random.default_rng().normal(price, sigma, volume), ticksize)
        v2p = collections.Counter(price_spread)
        vp=[(p, v2p[p]) for p in sorted(v2p.keys())]
        prices, volumes = zip(*vp)
        prices=np.array(prices)
        volumes=np.array(volumes)
        volumes[:len(volumes)//2] = volumes[:len(volumes)//2][::-1]
        volumes[len(volumes)//2:] = volumes[len(volumes)//2:][::-1]
        bidprices=prices[:len(prices)//2]
        bidvolumes=volumes[:len(volumes)//2]
        askprices=prices[len(prices)//2:]
        askvolumes=volumes[len(volumes)//2:]
        timestamp = time.time()
        yield {
            "timestamp": timestamp,
            "symbol":symbol,
            "askPrices":askprices,
            "askVolumes":askvolumes,
            "bidPrices":bidprices,
            "bidVolumes":bidvolumes
        }




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