#!python
#matching engine in python
from math import inf
import pickle

# We want to capture not only the order fills, but the order placements as well, along with their IDs
# The goal of these notifications is to create a live scrolling limit order book
# If an order is unfilled, we would see the place without the fill
# If an order is fully filled immediately, we would see both a place and fills, sum(fillqtys) == place qtys
# If an order is partially filled, we would see both place and fills, sum(fill qtys) != place qtys

class OrderBook:
    def __init__(self, savefile=None):
        self.bids=[] #stored highest to lowest in price
        self.asks=[] #stored lowest to highest in price
        self.savefile=savefile
        self.restore()
    def __del__(self):
        self.save()
    
    def save(self):
        if self.savefile is not None:
            with open(pickle, "rb") as fp:
                self.bids, self.asks = pickle.load(fp)

    def restore(self):
        if self.savefile is not None:
            with open(self.savefile, "wb") as fp:
                pickle.dump((self.bids, self.asks), fp)
        
    
    #  match_book: orders against; attempt to match against these
    #  add_book: orders on same side; add leftover liquidity here
    #  comp: (prc1, prc2) -> bool; returns true if prc1 is better than prc2
    #          when buying, higher is better; when selling, lower is better
    # RETURNS: a JSON formatted list of fills
    def process_order(self, id, qty, stop, limit, fok, alo, match_book, add_book):
        # (order) -> bool; returns true if new order is more attractively priced than given arg
        better = lambda offer: stop <= offer['prc'] < limit
        # (order) -> bool; matching consideration is same as better except equality also matches
        matches = lambda offer: better(offer) or offer['prc'] == limit

        place=[]
        prc=limit if add_book is self.bids else stop

        if alo: # Add Liquidity Only
            if len(match_book) > 0 and matches(match_book[0]):
                return { 'place': [], 'fills': [] }

        if fok: # Fill or Kill
            matching = filter(matches, match_book)
            matching_qty = sum(map(lambda offer: offer['qty'], matching))
            if matching_qty < qty:
                return { 'place': [], 'fills': [] }

        fills = []

        # match from the front of the book onwards, until price no longer crosses
        # continually examining the first element of match_book, fills are deleted from front as we iterate
        while qty > 0 and len(match_book) > 0:
            match = match_book[0]
            if not matches(match):
                break # match_book[1:] cannot match if match_book[0] doesn't, b/c match_book is ordered, best-first

            if match['qty'] <= qty: # top of book is exhausted
                fills.append(match)
                qty -= match['qty']
                del match_book[0]
            else: # top of book has remaining liquidity, new order is exhausted
                fills.append({'qty': qty, 'prc': match['prc'], 'id':match['id']})
                match['qty'] -= qty
                qty = 0

        # if there is liquidity remaining, find the right spot to insert to the book
        if qty > 0:
            # there should be a more pythonic way to do this search / insert
            idx = 0
            while idx < len(add_book):
                if better(add_book[idx]):
                    break
                idx += 1
            add_book.insert(idx, {'qty': qty, 'prc': prc, 'id':id})
            place.append({'qty':qty,'prc':prc, 'id':id})
        return { 'place':place, 'fills': fills }
    
    def cancel_order(self, id, qty, prc, book):
        cancels={'cancels':[]}
        for order in book:
            if order['id']==id and order['prc']==prc:
                order_qty=order['qty']
                if qty < order_qty:
                    order['qty'] -= qty
                    cancels['cancels'].append({'qty':qty,'prc':prc, 'id':id})
                    break
                else:
                    cancels['cancels'].append(order)
                    del order
                qty -= order_qty
        return cancels
    def cancel_all(self, id):
        cancels={'cancels':[]}
        for book in [self.bids, self.asks]:
            for order in book:
                if order['id'] == id:
                    cancels['cancels'].append(order)
                    del order
    def limitbuy(self, id, qty, prc, fok, alo):
        return self.process_order(id, qty, 0, prc, fok, alo, self.asks, self.bids)
    def limitsell(self, id, qty, prc, fok, alo):
        return self.process_order(id, qty, prc, inf, fok, alo, self.bids, self.asks)
    def marketbuy(self, id, qty):
        return self.process_order(id, qty, 0, inf, False, False, self.asks, self.bids)
    def marketsell(self, id, qty):
        return self.process_order(id, qty, 0, inf, False, False, self.bids, self.asks)
    def stoplimitbuy(self, id, qty, stop, limit, fok, alo):
        return self.process_order(id, qty, stop, limit, fok, alo, self.asks, self.bids)
    def stoplimitsell(self, id, qty, stop, limit, fok, alo):
        return self.process_order(id, qty, stop, limit, fok, alo, self.bids, self.asks)
    def cancelbuy(self, id, qty, prc, *argv):
        return self.cancel_order(id, qty, prc, self.bids)
    def cancelsell(self, id, qty, prc, *argv):
        return self.cancel_order(id, qty, prc, self.asks)
