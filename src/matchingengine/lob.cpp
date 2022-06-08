using namespace std;

#include "lob.h"
#include <unordered_map>
#include <chrono>

using namespace chrono;

//https://www.geeksforgeeks.org/c-program-red-black-tree-insertion/

/*
what happens if we add a bid order on the bidLimits tree, 
and then place a ask order on that same limit tree, even though the previous order was on the bid side?
We have 1 limit cache, so we can use that same one to find the right order
That limit was already in the bidTree, and now we do an add, we see the limit in the limitcache, so we won't 
add the limit to the selltree. The tree that a particular limit is in, is based on the side of the first order
on the list

*/

Book::Book(){
    _orderId=0;
}

uint64_t Book::add(Side side, uint64_t qty, uint64_t limitPrice, uint64_t time)
{
    auto evt_time = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    _orderId++;
    unordered_map<uint64_t,set<Limit>::iterator>::const_iterator it = limitCache.find(limitPrice);
    set<Limit>::iterator limitit;
    list<Order>::iterator orderit;
    bool s;

    if (it == limitCache.end()){//add to tree if not in cache (Ologn, sometimes O(1))
        if (side == BUYSIDE){
            if (bidLimits.empty()){
                auto[limitit,s]=bidLimits.emplace(limitPrice, side);
                maxBid=limitit;
            } else {
                if (limitPrice > maxBid->limitPrice){
                    limitit=bidLimits.emplace_hint(bidLimits.begin(), limitPrice, side); //O(1) for above max bid
                    maxBid=limitit;
                } else {
                    auto[limitit,s]=bidLimits.emplace(limitPrice,side); //O(logn) if 
                }
            }
        }
        else { //sell side
            if (askLimits.empty()){
                auto[limitit,s]=askLimits.emplace(limitPrice,side);
                minAsk=limitit;
            } else {
                if (limitPrice < minAsk->limitPrice){
                    limitit=askLimits.emplace_hint(askLimits.begin(), limitPrice, side);
                    minAsk=limitit;
                } else {
                    auto[limitit,s]=askLimits.emplace(limitPrice, side);
                }
            }
        }
        limitCache[limitPrice] = limitit;
    } else {
        limitit = it->second;
    }
    Limit *limit = const_cast<Limit *>(&(*limitit));
    orderit=limit->orders.emplace(limit->orders.end(),_orderId, side, qty, time, evt_time, limitit);
    orderCache[_orderId] = orderit; 

    return _orderId;
}


void Book::cancel(uint64_t orderId){
    unordered_map<uint64_t, list<Order>::iterator>::iterator it = orderCache.find(orderId);
    if (it == orderCache.end()){
        throw runtime_error("Invalid orderId");
    }
    list<Order>::iterator orderit = it->second;
    set<Limit>::iterator limitit=orderit->parentLimit;
    //const_cast is hacky and could lead to unsafe behavior if you modify the set comparison variables
    Limit *limit = const_cast<Limit *>(&(*limitit));


    if (limit->size() == 0){
        throw runtime_error("Empty order list");
    }

    limit->orders.erase(orderit);
    limit->totalVolume-=orderit->qty * limit->limitPrice;
    

    //if this is the last order on a limit, we want to delete it
    //to do this efficiently, if it's outside the minAsk maxBid limits,
    //then we can assign minAsk/maxBid to the next best limit
    // then/otherwise we will not delete here, we will delete in the limit_gc function
    if (limit->side == BUYSIDE && limit->limitPrice == minAsk->limitPrice){
        minAsk++;
    } else if (limit->side == SELLSIDE && limit->limitPrice == maxBid->limitPrice) {
        maxBid++;
    }

    orderCache.erase(it);
}

//this needs to be called periodically
//remove_if for const iterators is impossible with the current c++20 standard.
https://stackoverflow.com/questions/24263259/c-stdseterase-with-stdremove-if
void Book::limit_gc(){
    bidLimits.erase(remove_if(bidLimits.begin(), bidLimits.end(), [](Limit &limit){ return limit.size()==0; }));
    askLimits.erase(remove_if(askLimits.begin(), bidLimits.end(), [](Limit &limit){ return limit.size()==0; }));
}

//executes one trade. return the number of trades executes
// take the maxbuy and the minask, and work our way backwards

long Book::execute(){

    return 0;
}