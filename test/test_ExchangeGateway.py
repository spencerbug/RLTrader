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
    return 1

@pytest.fixture(scope="session")
def asset_btc(exchange_server):
    resp=requests.post(f"{exchange_server}/syncAdminApi/v1/assets", 
    json={"assetCode":"BTC","assetId":2,"scale":1000000})
    assert(resp.status_code//100==2)
    return 2

@pytest.fixture(scope="session")
def tradepair_btcusd(exchange_server, asset_usd, asset_btc):
    resp=requests.post(f"{exchange_server}/syncAdminApi/v1/symbols",
    json={
        "symbolCode":"BTCUSD",
        "symbolId":3,
        "symbolType":0,
        "baseAsset":"BTC",
        "quoteCurrency":"USD",
        "lotSize":1000000,
        "stepSize":10000,
        "takerFee":1900,
        "makerFee":700,
        "marginBuy": 0,
        "marginSell": 0,
        "priceHighLimit": 0,
        "priceLowLimit": 0
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

def placeOrder(exchange_server, userid, asset, amt, price, action, ordertype):
    resp=requests.post(f"{exchange_server}/syncTradeApi/v1/symbols/{asset}/trade/{userid}/orders/",
    json={
        "price":price,
        "size":amt,
        "userCookie": userid,
        "action": action,
        "orderType": ordertype
    })
    assert(resp.status_code//100==2)

def placeAskGTC(exchange_server, userid, asset, amt, price):
    placeOrder(exchange_server, userid, asset, amt, price, 0, 0)
    

def placeBidGTC(exchange_server, userid, asset, amt, price):
    placeOrder(exchange_server, userid, asset, amt, price, 1, 0)

def test_ipo_with_trades(exchange_server, tradepair_btc_usd, asset_usd, asset_btc):
    adduser(exchange_server, 1)
    adduser


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

