import pytest
from math import inf
from exchange.MatchingEngine import OrderBook

@pytest.fixture
def orderbook():
    return OrderBook()

@pytest.fixture
def FilledOrderBook(orderbook):
    orderbook.bids=[
        {'qty':5,'prc':5,'id':'a'},
        {'qty':5,'prc':4,'id':'b'},
        {'qty':5,'prc':3,'id':'c'},
        {'qty':5,'prc':2,'id':'d'},
        {'qty':5,'prc':1,'id':'e'}
    ]
    orderbook.asks=[
        {'qty':5,'prc':6,'id':'f'},
        {'qty':5,'prc':7,'id':'f'},
        {'qty':5,'prc':8,'id':'h'},
        {'qty':5,'prc':9,'id':'i'},
        {'qty':5,'prc':10,'id':'j'}
    ]
    return orderbook

def test_MatchingEngineInit(orderbook):
    assert orderbook.bids == []
    assert orderbook.asks == []


# For a buyer, asks should be listed low to high
# For a seller, bids should be listed high to low

def isInDescOrder(orders):
    prevOrderPrc=inf
    for order in orders:
        orderPrc=order['prc']
        assert orderPrc <= prevOrderPrc
        prevOrderPrc=orderPrc

def isInAscOrder(orders):
    prevOrderPrc=0
    for order in orders:
        orderPrc=order['prc']
        assert orderPrc >= prevOrderPrc
        prevOrderPrc=orderPrc

def test_BuyOrderPlacement(orderbook:OrderBook):
    orderbook.limitbuy('a',10, 8, False, False)
    orderbook.limitbuy('b',5, 7, False, False)
    orderbook.limitbuy('c',5, 12,False, False)
    isInDescOrder(orderbook.bids)

def test_SellOrderPlacement(orderbook:OrderBook):
    orderbook.limitbuy('a',10, 8, False, False)
    orderbook.limitbuy('b',5, 7, False, False)
    orderbook.limitbuy('c',5, 12,False, False)
    isInAscOrder(orderbook.asks)

def test_PlaceLimitSell(FilledOrderBook:OrderBook):
    ret=FilledOrderBook.limitsell('q', 6, 4, False, False)
    assert ret=={'place':[], 'fills':[{'qty':5,'prc':5, 'id':'a'},{'qty':1,'prc':4,'id':'b'}]}
    assert len(FilledOrderBook.bids) == 4
    assert len(FilledOrderBook.asks) == 5

def test_PlacePartialFilledLimitSell(FilledOrderBook:OrderBook):
    ret=FilledOrderBook.limitsell('q', 7, 5, False, False)
    assert ret=={'place':[{'id':'q','qty':2,'prc':5}],'fills':[{'qty':5,'prc':5,'id':'a'}]}
    assert len(FilledOrderBook.asks) == 6
    assert len(FilledOrderBook.bids) == 4

def test_PlaceUnfilledLimitBuy(FilledOrderBook:OrderBook):
    ret=FilledOrderBook.limitbuy('q', 5, 4, False, False)
    assert ret=={'place':[{'id':'q','qty':5,'prc':4}], 'fills':[]}
    assert len(FilledOrderBook.bids) == 6
    assert len(FilledOrderBook.asks) == 5

def test_PlaceLimitBuy(FilledOrderBook:OrderBook):
    fills=FilledOrderBook.limitbuy('q', 6, 7, False, False)
    assert fills=={'fills':[{'qty':5, 'prc':6, 'id':'f'},{'qty':1,'prc':7, 'id':'f'}], 'place':[]}
    assert len(FilledOrderBook.bids) == 5
    assert len(FilledOrderBook.asks) == 4

def test_CancelUnfilledLimitSell(FilledOrderBook:OrderBook):
    cancels=FilledOrderBook.cancelsell('f', 3,6)
    assert cancels=={'cancels':[{'qty':3,'prc':6,'id':'f'}]}
    assert FilledOrderBook.asks[0]['qty']==2


def test_PlaceMarketSell(FilledOrderBook:OrderBook):
    fills = FilledOrderBook.marketsell('q',9)
    assert fills=={'fills':[
        {'qty':5,'prc':5, 'id':'a'},
        {'qty':4,'prc':4, 'id':'b'},
    ], 'place':[]}
    assert len(FilledOrderBook.bids) == 4

def test_PlaceMarketBuy(FilledOrderBook:OrderBook):
    fills = FilledOrderBook.marketbuy('q',9)
    assert fills=={'fills':[
        {'qty':5,'prc':6,'id':'f'},
        {'qty':4,'prc':7,'id':'f'},
    ], 'place':[]}
    assert len(FilledOrderBook.asks) == 4
    