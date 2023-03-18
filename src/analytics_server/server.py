
# plan:
# use zeromq to receive messages from exchange, process the analytics, and publish them to the subscribers

import zmq
import requests
from analytics import fixeddepthchart_gen, ta_gen
from autoencoder import TimeSeriesAutoencoder, train_and_predict
import numpy as np
from itertools import tee
from numpy_ringbuffer import RingBuffer

# Set up the ZeroMQ context
context = zmq.Context()

# Set up the PUB socket for broadcasting
broadcast_socket = context.socket(zmq.PUB)
broadcast_socket.bind("tcp://*:5555")  # Change this to the desired address and port

EXCHANGE_SERVER="http://localhost:8080"


def query_fetcher(symbol):
    while True:
        respL2=requests.get(f"{EXCHANGE_SERVER}/syncTradeApi/v1/symbols/{symbol}/orderbook?depth=-1")
        respOHLCV=requests.get(f"{EXCHANGE_SERVER}/syncTradeApi/v1/symbols/{symbol}/bars/M1?count=1")
        try:
            L2=respL2.json()
            OHLCV=respOHLCV.json()
            yield {"L2":L2['data'], "OHLCV":OHLCV['data'][0]}
        except:
            yield {"L2":None, "OHLCV":None}

def get_gens_from_dict(gen, **kwargs):
    def make_gen(key):
        return (d[key] for d in gen)
    return tuple(make_gen(key) for key in kwargs)


def main():
    stepsize=0.01
    history_length=100
    symbol='BTCUSDT'
    depthchart_buckets=[1,2,3,4,5,6,7,8,10,13,17,21,27,34,44,55,72,89,100]
    datawidth=15 + (2*len(depthchart_buckets))

    rb = RingBuffer(history_length, dtype=(float, datawidth))

    autoencoder = TimeSeriesAutoencoder(100, 100, )

    ### Pipeline ###
    psource = query_fetcher(symbol)
    L2p0, OHLCVp0 = get_gens_from_dict(psource, L2gen="L2", OHLCVgen="OHLCV")
    OHLCVp1, OHLCVp2 = tee(OHLCVp0)
    FDCg = fixeddepthchart_gen(L2p0, stepsize, depthchart_buckets)
    TAg = ta_gen(OHLCVp1, 100)
    ### Pipeline ###

    for fd, ta, ohlc in zip(FDCg, TAg, OHLCVp2):
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

        # extract features from the time series data into an embedding vector using a pretrained model
        # send the embedding vector to the subscriber

