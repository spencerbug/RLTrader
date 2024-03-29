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

orderId_t Book::add(userId_t userId, Side side, uint64_t qty, price_t limitPrice, uint64_t time)
{
    auto evt_time = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    _orderId++;
    unordered_map<uint64_t, map<uint64_t, Limit>::iterator>::const_iterator it = limitCache.find(limitPrice);
    map<uint64_t, Limit>::iterator limitit;
    list<Order>::iterator orderit;
    bool s;

    if (it == limitCache.end()){//add to tree if not in cache (Ologn, sometimes O(1))
        if (side == BUYSIDE){
            if (bidLimits.empty()){
                auto[limitit,s]=bidLimits.emplace(limitPrice, side);
                maxBid=limitit;
            } else {
                if (limitPrice > maxBid->first){
                    limitit=bidLimits.emplace_hint(bidLimits.begin(), limitPrice, side); //O(1) for above max bid
                    maxBid=limitit;
                } else {
                    auto[limitit,s]=bidLimits.emplace(limitPrice, side); //O(logn) if 
                }
            }
        }
        else { //sell side
            if (askLimits.empty()){
                auto[limitit,s]=askLimits.emplace(limitPrice, side);
                minAsk=limitit;
            } else {
                if (limitPrice < minAsk->first){
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
    Limit& limit=limitit->second;
    limit.totalVolume+=qty * limitit->first;
    orderit=limit.orders.emplace(limit.orders.end(), userId, _orderId, side, qty, time, evt_time, limitit);
    orderCache[_orderId] = orderit; 

    return _orderId;
}


void Book::cancel(uint64_t orderId){
    unordered_map<uint64_t, list<Order>::iterator>::iterator it = orderCache.find(orderId);
    if (it == orderCache.end()){
        throw runtime_error("Invalid orderId");
    }
    list<Order>::iterator orderit = it->second;
    map<uint64_t, Limit>::iterator limitit=orderit->parentLimit;
    Limit& limit=limitit->second;


    if (limit.size() == 0){
        throw runtime_error("Empty order list");
    }

    limit.orders.erase(orderit);
    limit.totalVolume-=orderit->qty * limitit->first;
    

    //if this is the last order on a limit, we want to delete it
    //to do this efficiently, if it's outside the minAsk maxBid limits,
    //then we can assign minAsk/maxBid to the next best limit
    // then/otherwise we will not delete here, we will delete in the limit_gc function
    if (limit.side == BUYSIDE && limitit->first == minAsk->first){
        minAsk++;
    } else if (limit.side == SELLSIDE && limitit->first == maxBid->first) {
        maxBid++;
    }

    orderCache.erase(it);
}

//this needs to be called periodically
//remove_if for const iterators is impossible with the current c++20 standard.
//https://stackoverflow.com/questions/24263259/c-stdseterase-with-stdremove-if
// this function deletes limits with empty quantities.
void Book::limit_gc(bool outside_only){
    if (outside_only){
        for (auto it = bidLimits.begin(); it != bidLimits.end(); ){
            if ( it->second.size()==0  ){
                it=bidLimits.erase(it);
            } else {
                break;
            }
            it++;
        }
        for (auto it = bidLimits.rbegin(); it != bidLimits.rend(); ){
            if ( it->second.size()==0  ){
                advance(it,1);
                bidLimits.erase(it.base());
            } else {
                break;
            }
            it++;
        }
        for (auto it = askLimits.begin(); it != askLimits.end(); ){
            if ( it->second.size()==0  ){
                it=askLimits.erase(it);
            } else {
                break;
            }
            it++;
        }
        for (auto it = askLimits.rbegin(); it != askLimits.rend(); ){
            if ( it->second.size()==0  ){
                advance(it,1);
                askLimits.erase(it.base());
            } else {
                break;
            }
            it++;
        }    
    } else {
        for (auto it = bidLimits.begin(); it != bidLimits.end(); ){
            if ( it->second.size()==0  ){
                it=bidLimits.erase(it);
            } else ++it;
        }
        for (auto it = askLimits.begin(); it != askLimits.end(); ){
            if ( it->second.size()==0  ){
                it=askLimits.erase(it);
            } else ++it;
        }
    }
}

optional<map<uint64_t, Limit>::iterator> Book::getMatchLimit(list<Order>::iterator order){
    if (order->side == BUYSIDE){
        //if you're buying above the minAsk
        if (order->parentLimit->first >= minAsk->first){
            return minAsk;
        } else {
            return nullopt;
        }
    } else {
        //if you're selling below maxBid
        if (order->parentLimit->first <= maxBid->first){
            return maxBid;
        } else {
            return nullopt;
        }
    }
}

//checks if there are any available matches in the lob.
bool Book::can_match(){
    return minAsk->first <= maxBid->first;
}

//executes one trade. return the number of trades executes
// take the maxbuy and the minask, and work our way backward
long Book::execute(){
    if (can_match){
        //offerer is first come first serve (time), which is matched to a group of offerees until the offerer's qty is used up
        list<Order>::iterator firstMinAsk = minAsk->second.orders.begin();
        list<Order>::iterator firstMaxBid = maxBid->second.orders.begin();
        //the first buyer or seller we match is the earliest of either the minAsk or maxBid
        Side whosfirst = firstMaxBid->entryTime < firstMinAsk->entryTime ? BUYSIDE : SELLSIDE;
        if (whosfirst==BUYSIDE){
            list<Order>::iterator offerer=firstMinAsk;
            optional<map<uint64_t, Limit>::iterator> offereeLimit = getMatchLimit(offerer);
            
        }
    }
    return 0;
}