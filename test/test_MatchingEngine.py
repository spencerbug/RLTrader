import pytest

from exchange import MatchingEngine as me

@pytest.fixture
def OrderBook():
    return me.OrderBook()

def test_MatchingEngineInit(OrderBook):
    assert OrderBook.bids == []
    assert OrderBook.asks == []
