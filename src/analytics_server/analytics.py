# this server will connect to the exchange and 
# get the limit order book data in realtime
# and will convert that into RLTrader readable information

from time import time
import requests
import numpy as np
from numpy import nan as NaN
from numbers import Number as NumberType
from numpy_ringbuffer import RingBuffer
from talib import stream as tastream
import talib



class CircularBuffer:
    def __init__(self, size=100):
        self.size=size
        self.buf=[None]*size
        self.head=0
        self.tail=0
        self.count=0
        self.sum=0
    
    def isFull(self):
        return (self.count == self.size)
    
    def isEmpty(self):
        return self.count == 0
    
    def __len__(self):
        return self.count
    
    def avg(self):
        return self.sum / self.count
    
    def add(self, item):
        if self.isFull():
            # if we are full, we start eating our own tail. Shrink the tail
            self.head=(self.head+1) % self.size
        else:
            self.count += 1
        olditem=self.buf[self.tail]
        if olditem is not None and isinstance(olditem, NumberType):
            self.sum -= olditem
        if isinstance(item, NumberType):
            self.sum += item
        self.buf[self.tail] = item
        self.tail = (self.tail+1) % self.size
    
    def __iter__(self):
        idx = self.head
        num = self.count
        while num > 0:
            yield self.buf[idx]
            idx = (idx + 1) % self.size
            num -= 1
    
    def __repr__(self):
        # String representation of the circular buffer
        if self.isEmpty():
            return "cb:[]"
        else:
            return 'cb:[' + ','.join(map(str,self)) + ']'


"""
fixeddepthchart_gen
builds a depth chart of a fixed size

Args:
* stepsize: is the minimum change in price
* bucketindices: is an iter or list that represents the "spread" of how volumes are accumulated with respect to price(=index*stepssize) from the center
  volume of each bucket = total volume with price range of: offsetprice += (bucketindices[i-1]*stepsize, bucketindices[i]*stepsize)
* cumulative: if true, the volume at each pricebucket is cumulative looking more like a traditional depth chart

Yield:
outputs a dict {"bidPrices":bidPrices,"bidVolumes":bidVolumes, "askPrices":askPrices, "askVolumes":askVolumes}

lets say buckindices=[1,2,4,7,10]
stepsize=1
orderbook = {
    "bidPrices": [1,4,5.5,6,7,13.5],
    "bidVolumes":[2,1,1,  3,1,2],
    "askPrices": [12,13.5,15,17],
    "askVolumes":[1, 2,   4, 3]
}

The output should be a vector of size len(bucketindices)*2
The bucketindices represent a mirror image so they are symetrical:
bidprice: 10,7,4,2,1,
askprice: 1,2,4,7,10
The pricebucket numbers represent all the volumes in a range price indices:

bidprice_idx_ranges: (10)-7, (7)-4, (4)-2, (2)-1, (1)-0
askprice_idx_ranges: 0-(1), 1-(2), 2-(4), 4-(7), 7-(10)
a-(b) means a up to but not including b

Now we translate these price range indices into actual prices
first calculate the offset:
maxBid =max(bidprices)=13.5
minAsk =min(askprices)=12

then we apply the formula offset +- p*stepsize
for bidprice its maxBid - bidprice_idx_range*stepsize
for askprice its minAsk + askprice_idx_range*stepsize

Then we go through each of the ranges and accumulate the volumes at those prices


"""
def fixeddepthchart_gen(orderbook_gen, stepsize=0.01, bucketindices=[1,2,3,4,5,6,7,8,10,13,17,21,27,34,44,55,72,89, 100], cumulative=False):
    for orderbook in orderbook_gen:
        bidPrices = orderbook["bidPrices"]
        bidVolumes = orderbook["bidVolumes"]
        askPrices = orderbook["askPrices"]
        askVolumes = orderbook["askVolumes"]

        # output buffer for bids
        outputBidVolumes=np.zeros((len(bucketindices)))

        # output buffer for asks
        outputAskVolumes=np.zeros((len(bucketindices)))

        outputBidPriceRanges=np.zeros((len(bucketindices),2))
        outputAskPriceRanges=np.zeros((len(bucketindices),2))

        # pad bucketindices with 0
        if bucketindices[0] != 0:
            bi = [0] + bucketindices
        

        # accumulate bidprices
        if len(bidPrices) > 0:
            outputBidPriceRanges=np.zeros((len(bucketindices),2))
            outputBidVolumes=np.zeros((len(bucketindices)))
            bktptr=0
            prcptr=len(bidPrices)-1
            maxBid=bidPrices[-1]
            while bktptr < (len(bi) - 1):
                priceWindowA=max(maxBid - (bi[bktptr + 1] * stepsize),0)
                priceWindowB=maxBid - (bi[bktptr] * stepsize)
                if prcptr >= 0 and prcptr < len(bidPrices) and priceWindowA < bidPrices[prcptr] and bidPrices[prcptr] <= priceWindowB:
                    outputBidVolumes[-(1+bktptr)] += bidVolumes[prcptr]
                    prcptr -= 1
                else:
                    outputBidPriceRanges[-(1+bktptr)] = [priceWindowA, priceWindowB]
                    bktptr += 1
                    if cumulative and bktptr > 0:
                        outputBidVolumes[bktptr] += outputBidVolumes[bktptr - 1]
        

        # accumulate askprices
        if len(askPrices) > 0:
            outputAskPriceRanges=np.zeros((len(bucketindices),2))
            outputAskVolumes=np.zeros((len(bucketindices)))
            bktptr=0
            prcptr=0
            minAsk=askPrices[0]
            while bktptr < (len(bi) - 1):
                priceWindowA=minAsk + (bi[bktptr] * stepsize)
                priceWindowB=minAsk + (bi[bktptr + 1] * stepsize)
                if prcptr < len(askPrices) and priceWindowA <= askPrices[prcptr] and askPrices[prcptr] < priceWindowB:
                    outputAskVolumes[bktptr] += askVolumes[prcptr]
                    prcptr += 1
                else:
                    outputAskPriceRanges[bktptr] = [priceWindowA, priceWindowB]
                    bktptr += 1
                    if cumulative and bktptr > 0:
                        outputAskVolumes[bktptr] += outputAskVolumes[bktptr - 1]
        
        yield {
            "bidPrices":outputBidPriceRanges,
            "bidVolumes":outputBidVolumes,
            "askPrices":outputAskPriceRanges,
            "askVolumes":outputAskVolumes
        }
        



"""
ta_gen builds technical analysis parameters

PARAMS
bars_gen: A yield generator ouputting 6 parameters as a dictionary: 'open', 'high', 'low', 'close', 'volume', 'timestamp'
capacity: size for ring buffer. Acts as TA window.

OUTPUT:
yields a dictionary: 'macd', 'adx', 'sma', 'rsi', 'obv', 'wvap', 'cci', 'tsi'
moving average convergence-divergence
average directional index
simple moving average
relative strength index
on-balance vaolume
weighted volume average price
commodity channel index
true strength index
"""
def ta_gen(bars_gen, capacity=100):
    rb:RingBuffer = RingBuffer(capacity, dtype=(float, 6))
    OPEN=0; HIGH=1; LOW=2; CLOSE=3; VOLUME=4; TIMESTAMP=5
    for bars in bars_gen:
        rb.append([bars["open"], bars["high"], bars["low"], bars["close"], bars["volume"], bars["timestamp"]])
        macd = tastream.MACD(rb[:,CLOSE], fastperiod=12, slowperiod=26, signalperiod=9) # returns 3 values: macd, signal, and divergence (https://en.wikipedia.org/wiki/MACD)
        adx = tastream.ADX(rb[:,HIGH], rb[:,LOW], rb[:, CLOSE], timeperiod=14)
        sma = tastream.SMA(rb[:,CLOSE], timeperiod=30)
        rsi = tastream.RSI(rb[:,CLOSE], timeperiod=14)
        obv = tastream.OBV(rb[:,CLOSE], rb[:,VOLUME])
        wvap = WVAP(rb[:,VOLUME], rb[:,HIGH], rb[:,LOW], rb[:,CLOSE], stream=True)
        cci = tastream.CCI(rb[:,HIGH], rb[:,LOW], rb[:,CLOSE], timeperiod=14)
        tsi = TSI(rb[:,CLOSE], long_tp=25, short_tp=13, stream=True)

        yield {
            "macd":macd[0],
            "macd_sig":macd[1],
            "macd_div":macd[2],
            "adx":adx,
            "sma":sma, 
            "rsi":rsi, 
            "obv":obv, 
            "wvap":wvap, 
            "cci":cci, 
            "tsi":tsi
        }

# weighted volume average price
# 'stream mode' sets timeperiod to 1, and outputs a single number not an array
def WVAP(volume:np.ndarray, high:np.ndarray, low:np.ndarray, close:np.ndarray, timeperiod=30, stream=False):
    if stream:
        timeperiod=1
    typical=(high[-timeperiod:] + low[-timeperiod:] + close[-timeperiod:])/3
    wvap = np.divide(np.multiply(volume[-timeperiod:], typical), np.sum(volume[-timeperiod:]))
    if stream:
        return wvap[0]
    return wvap


# true strength indicator
def TSI(close:np.ndarray, long_tp=25, short_tp=13, stream=False):
    try:
        TSI.counter = TSI.counter+1
    except AttributeError:
        TSI.counter = 0
    assert(long_tp > short_tp)
    if len(close) < (long_tp+short_tp):
        return NaN
    if stream:
        pc = np.diff(close[-(long_tp+short_tp):]) #stream mode calculated just the most recent element
    else:
        pc = np.diff(close) # subtracts each close price with previous close prie
    pcs = talib.EMA(pc, timeperiod=long_tp)
    pcds = talib.EMA(pcs, timeperiod=short_tp)
    apc = np.abs(pc)
    apcs = talib.EMA(apc, timeperiod=long_tp)
    apcds = talib.EMA(apcs, timeperiod=short_tp)
    tsi = np.divide(pcds,apcds)
    return tsi[-1]


def combine_and_fill_in_sparse_price_volumes(orderbook_gen, stepsize=0.01):
    for orderbook in orderbook_gen:
        sparse_prices=[*orderbook['bidPrices'], *orderbook['askPrices']]
        # let's make ask volumes negative
        sparse_volumes=[*orderbook['bidVolumes'], *np.multiply(orderbook['askVolumes'], -1)]

        # assume price is sorted, and len(prices)==len(volumes)
        assert len(sparse_prices) == len(sparse_volumes)

        # fill in zeros in the empty stepsized spaces in prices and volumes
        # such that every price is accounted for, even the ones with no order volume
        newsize=round((sparse_prices[-1] - sparse_prices[0]) / stepsize) + 1
        prices= np.arange(sparse_prices[0], sparse_prices[-1], stepsize)
        volumes=np.zeros(newsize)
        # now fill in the values into from the sparse array
        # i=0,1,2,3
        for i in range(0,len(sparse_prices)):
            newidx = round((sparse_prices[i]-sparse_prices[0])/stepsize)
            volumes[newidx] = sparse_volumes[i]
        
        orderbook['prices']=prices
        orderbook['volumes']=volumes
        yield orderbook
