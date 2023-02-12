from random import randint
from time import sleep
import pytest
import requests
import subprocess
from conftest import get_git_root

@pytest.fixture(scope="session")
def exchange_server():
    p=subprocess.Popen("docker-compose up", cwd=get_git_root())
    sleep(10)
    yield "localhost:8080"
    p.terminate()

@pytest.fixture(scope="session")
def asset_usd(exchange_server):
    resp=requests.post(f"{exchange_server}/syncAdminApi/v1/assets", 
    json={"assetCode":"USD","assetId":1,"scale":1})
    assert(resp//100==2)
    return "USD"

@pytest.fixture(scope="session")
def asset_btc(exchange_server):
    resp=requests.post(f"{exchange_server}/syncAdminApi/v1/assets", 
    json={"assetCode":"BTC","assetId":2,"scale":1000000})
    assert(resp.status_code//100==2)
    return "BTC"

@pytest.fixture(scope="session")
def tradepair_btcusd(exchange_server, asset_usd, asset_btc):
    resp=requests.post(f"{exchange_server}/syncAdminApi/v1/symbols",
    json={
        "symbolCode":"BTCUSD",
        "symbolId":3,
        "symbolType":0,
        "baseAsset":asset_btc,
        "quoteCurrency":asset_usd,
        "lotSize":0.1,
        "stepSize":0.01,
        "takerFee":0.08,
        "makerFee":0.03,
        "marginBuy": 0,
        "marginSell": 0,
        "priceHighLimit": 50000,
        "priceLowLimit": 1000
    })
    assert(resp.status_code//100==2)
    return "BTCUSD"

def adduser(exchange_server, id):
    resp=requests.post(f"{exchange_server}/syncAdminApi/v1/users",json={"id":id})
    assert(resp.status_code//100==2)
    return id

def addbalance(exchange_server, userid, asset, amt):
    if not hasattr(addbalance, "tid"):
        addbalance.tid=0
    resp=requests.post(f"{exchange_server}/syncAdminApi/v1/users/{userid}/accounts",
    json={
        "transactionId":addbalance.tid,
        "amount":amt,
        "currency":asset
    })
    assert(resp.status_code//100==2)
    addbalance.tid += 1

def placeOrder(exchange_server, userid, symbol, amt, price, action, ordertype):
    resp=requests.post(f"{exchange_server}/syncTradeApi/v1/symbols/{symbol}/trade/{userid}/orders/",
    json={
        "price":price,
        "size":amt,
        "userCookie": userid,
        "action": action,
        "orderType": ordertype
    })
    assert(resp.status_code//100==2)

def placeAskGTC(exchange_server, userid, symbol, amt, price):
    placeOrder(exchange_server, userid, symbol, amt, price, 0, 0)
    

def placeBidGTC(exchange_server, userid, symbol, amt, price):
    placeOrder(exchange_server, userid, symbol, amt, price, 1, 0)

def test_ipo_with_trades(exchange_server, tradepair_btc_usd, asset_usd, asset_btc):
    adduser(exchange_server, 1)
    adduser(exchange_server, 2)
    adduser(exchange_server, 3)
    addbalance(exchange_server, 1, asset_btc, 1000)
    addbalance(exchange_server, 2, asset_usd, 10000)
    addbalance(exchange_server, 3, asset_usd, 10000)
    placeAskGTC(exchange_server, 1, tradepair_btc_usd, 1000, 2000.0)
    placeBidGTC(exchange_server, 2, tradepair_btc_usd, 999, 2000.0)
    placeAskGTC(exchange_server, 2, tradepair_btc_usd, 998, 2002.0)
    placeBidGTC(exchange_server, 3, tradepair_btc_usd, 3, 2002.0)
    

# We are trying to get trades going to fake out some sample data
# So steps would be create two assets, usd, btc.
# Then create a tradepair
# Create the 'exchange' user which will facilitate the ipo
# Add a balance to the exchange user of btc
# Offer up the amount of btc for sale in exchange for usd
# Create the regular user, add a balance of usd
# pick a price at or above the market, and buy the btc.
# turn it around and sell it 
# rinse and repeat for a few more buyer and sellers to start to get some rebounding effects and price discovery

