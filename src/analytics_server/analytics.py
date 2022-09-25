# this server will connect to the exchange and 
# get the limit order book data in realtime
# and will convert that into RLTrader readable information

from time import time
import requests
from sklearn.utils import resample
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import cheby1, lfilter
import numpy as np
from numbers import Number as NumberType


def get_order_book():
    resp=requests.get("localhost:8080/syncTradeApi/v1/symbols/BTCUSD/orderbook?depth=100")
    return resp.json()

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

        
        

# first step is to resample the order book to a fixed size
# this is easier said than done
# the order book volumes are at sparse prices, without any price bounds
# first step: fill in the in-between prices with 0 volume values.
# second step: setermine a sensible upper and lower bounds for the volumes based on volatility, and crop
# third step: to scale the cropped volumes to fixed price size. If scaling down we may need anti-aliasing
# YIELD value:
# the result will be x,y values, 
# where x is an array of prices size pricebuckets, centered and scaled around the point of volatility
# the y is an array of volumes, including the in-between values

def normalize_order_book(gen, timewindow=100, pricebuckets=1000, stepsize=0.01):
    #volatility buffer
    vtybuf = CircularBuffer(timewindow)
    
    #fetch a snapshot of the orderbook
    for orderbook in gen:
        sparse_prices=[*orderbook.bidPrices, *orderbook.askPrices]
        # let's make ask volumes negative
        sparse_volumes=[*orderbook.bidVolumes, *np.multiply(orderbook.askVolumes, -1)]
        
        # assume price is sorted, and len(prices)==len(volumes)
        assert len(sparse_prices) == len(sparse_volumes)

        # fill in zeros in the empty stepsized spaces in prices and volumes
        # such that every price is accounted for, even the ones with no order volume
        newsize=(sparse_prices[-1] - sparse_prices[0]) / stepsize
        prices= np.linspace(sparse_prices[0], sparse_prices[1], stepsize)
        volumes=np.zeros(newsize)
        # now fill in the values into from the sparse array
        for i in range(0,len(sparse_prices)):
            newidx = int((sparse_prices[i]-sparse_prices[0])/stepsize)
            volumes[newidx] = sparse_volumes[i]

        # add midprice to circular buffer
        # assume bid and ask prices are sorted ascending order
        midprice = (orderbook.bidPrices[-1] + orderbook.askPrices[0])/2
        # track the midprice over time each fetch
        vtybuf.add(midprice)

        # moving average and moving variance of circular buffer
        mu = np.mean(vtybuf)
        sigma = np.sqrt( (np.sum(vtybuf - mu)**2) / len(vtybuf) )

        # 3 sigma above and below our midprice will be the limits. cut out data outside that range
        pricewindow_lo=midprice - (3*sigma)
        pricewindow_hi=midprice + (3*sigma)

        # slice a new array from our window
        leftcrop = next((i for i,x in enumerate(prices) if x > pricewindow_lo))
        rightcrop = next((i for i,x in enumerate(reversed(prices)) if x < pricewindow_hi))
        cropped_prices = prices[leftcrop:rightcrop]
        cropped_volumes = volumes[leftcrop:rightcrop]


        # we are going to rescale the volumes to a new price window size
        oldwindowsize=(rightcrop - leftcrop)
        newwindowsize = pricebuckets
        
        if ( newwindowsize < oldwindowsize/2 ): # scaling down
            # if scaling down to less than half we need to apply an anti-aliasing filter
            old_fs = 1/stepsize
            new_fs = old_fs * newwindowsize / oldwindowsize
            a, b = cheby1(N=4, rp=3, Wn=new_fs, fs=old_fs, output='ba', analog=False, btype='lowpass')
            newvolumes = lfilter(b, a, cropped_volumes)
            # $a_0 y[n] = b_0 x[n] + b_1 x[n-1] + b_2 x[n-2] - a_1 y[n-1] - a_2 y[n-2]
            f = interp1d(cropped_prices, newvolumes)
        else: # scaling up
            f = interp1d(cropped_prices, cropped_volumes)
            
        # now that we have our linear function f we can generate our fixed bucket volumes
        x = np.arange(cropped_prices[0], cropped_prices[-1], pricebuckets)
        y = [f(xi) for xi in x]

        yield x,y



    # // get volumetric bid ask spread, except we want to normalize to a fixed window
    # //a naive approach for this would be to scale the volumetric data to 1000 values
    # // we could track the min and max limit prices over a moving window, and scale to that.
    # //the problem is if one user decides to put their limit wayyy up high, it will scale out the data instantly,
    # //which would mess with the graph
    # // to deal with outliers we have two options:
    # //1) limit the order prices to a certain percentage above and below market price and use z-score normalization
    # //2) use robust data scaling and ignore outliers, by calculating the median
    # // I'm going to go with option 1. which means we can just regular standardization
    # //taking each limit price:
    # // LP' = (LP - avg LP) / stdev(LP)
    # //here are the limit prices over 3 time points. Let's say our resolution is 10 buckets
    # //1,2,5,6,10
    # //1,5,6
    # //4,6,9,11
    # //6,7,9,10,11
    # //lets say our window resolution is 3. So we take the max and limits of a moving window of 3
    # //so for 0,1,2 min is 1, max is 11. So we need to fit limits 1-11 in buckets of range 0-10
    # /*
    # first we need to find a moving min/max window using deques. (largest max and smallest min over N sets of limits)
    # then we need to map our window to those min/max limits
    # lets say we have 1000 buckets in our window
    # min if 3431, max is 3478
    # we would then need to upsample or downsample (in this case upsample) to fit our new range 
    # We can use linear interpolation to scale x data (limit price) and y data (volume)
    # We iterate from limitMin to limitMax, and any missing limits are filled as 0 volume
    # upsampling algo:
    # a=3431, b=3432
    # from i=0to1000 (our window)
    #     x = remap i to range 3431...3478 (double)
    #     if x>b: a++, b++
    #     y=lerp(x,a,b, limit[a],limit[b])

    # downsampling algo:
    # we can use a lowpass filter (chebyshev type 1)
    # decimation
    # lets say min is 3431 and max is 3478 (47 points) and we want to downsample to 20 buckets

    # Online from what I've read you have to upsample, LP filter, then downsample, to get a fraction of the upsample/downsample
    # to resize the signal
    # We can use polyphase arbitrary resampler!
    
    