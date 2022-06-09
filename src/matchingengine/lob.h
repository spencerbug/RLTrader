//limit order book
// reference: https://gist.github.com/halfelf/db1ae032dc34278968f8bf31ee999a25
// https://github.com/Kautenja/limit-order-book/blob/master/notes/lob.md

#ifndef _LOB_H
#define _LOB_H

#include <unordered_map>
#include <map>
#include <list>
#include <vector>
#include <limits>

using namespace std;
typedef enum {SELLSIDE, BUYSIDE} Side;

class Order {
public:
    const uint64_t orderId;
    Side side;
    uint64_t qty;
    uint64_t limit;
    uint64_t entryTime;
    uint64_t eventTime;
    map<uint64_t, Limit>::iterator parentLimit;
    Order(uint64_t orderId, Side side, uint64_t qty, uint64_t entryTime, uint64_t eventTime, map<uint64_t, Limit>::iterator parentLimit):
    orderId(orderId), side(side), qty(qty), limit(limit), entryTime(entryTime), eventTime(eventTime), parentLimit(parentLimit)
    {}
};

class Limit {
    public:
    // const uint64_t limitPrice;  //limitPrice is iterator->first (map key)
    const Side side;
    uint64_t totalVolume; //sum(order.qty*price)
    list<Order> orders;
    Limit(Side side): side(side) {};
    const size_t size(){
        return this->orders.size();
    }
};

//RBTree and hashmap combo for limits

class Book {
    map<uint64_t, Limit, greater<uint64_t>> bidLimits;
    map<uint64_t, Limit, less<uint64_t>> askLimits;

    map<uint64_t, Limit>::iterator minAsk, maxBid;
    unordered_map<uint64_t, map<uint64_t, Limit>::iterator> limitCache; //maps limit price to Limit
    unordered_map<uint64_t, list<Order>::iterator> orderCache; //maps orderId to Order
    uint64_t _orderId;
public:

    //places order at the end of a list of orders to be execute at a limit price
    //O(log M) for first order to add at a limit, O(1) for others. Using a binary tree and a hash(of limit) table for that
    uint64_t add(Side side, uint64_t qty, uint64_t limit, uint64_t time);

    //removes an arbitrary order
    void cancel(uint64_t orderId); // O(1) h

    //removes an order from inside the book. "inside" is oldest buy at highest price and oldest sell at lowest price
    long execute(); //performs optimal execution between a buy and sell side

    // gets the volume at a given limit
    uint64_t getVolumeAtLimit(Side side, uint64_t limit); //O(1)

   //prints the volumes at each limit price (sparse)
    vector<vector<uint64_t>>& get_book_volumes();

    // gets the optimal order for a given limit
    Order *getBestBidOrOffer(uint64_t limit, Side side); // O(1)

    // delete empty limits in the buy and sell trees
    void limit_gc(); 

};

#endif

/*
    // get volumetric bid ask spread, except we want to normalize to a fixed window
    //a naive approach for this would be to scale the volumetric data to 1000 values
    // we could track the min and max limit prices over a moving window, and scale to that.
    //the problem is if one user decides to put their limit wayyy up high, it will scale out the data instantly,
    //which would mess with the graph
    // to deal with outliers we have two options:
    //1) limit the order prices to a certain percentage above and below market price and use z-score normalization
    //2) use robust data scaling and ignore outliers, by calculating the median
    // I'm going to go with option 1. which means we can just regular standardization
    //taking each limit price:
    // LP' = (LP - avg LP) / stdev(LP)
    //here are the limit prices over 3 time points. Let's say our resolution is 10 buckets
    //1,2,5,6,10
    //1,5,6
    //4,6,9,11
    //6,7,9,10,11
    //lets say our window resolution is 3. So we take the max and limits of a moving window of 3
    //so for 0,1,2 min is 1, max is 11. So we need to fit limits 1-11 in buckets of range 0-10
    /*
    first we need to find a moving min/max window using deques. (largest max and smallest min over N sets of limits)
    then we need to map our window to those min/max limits
    lets say we have 1000 buckets in our window
    min if 3431, max is 3478
    we would then need to upsample or downsample (in this case upsample) to fit our new range 
    We can use linear interpolation to scale x data (limit price) and y data (volume)
    We iterate from limitMin to limitMax, and any missing limits are filled as 0 volume
    upsampling algo:
    a=3431, b=3432
    from i=0to1000 (our window)
        x = remap i to range 3431...3478 (double)
        if x>b: a++, b++
        y=lerp(x,a,b, limit[a],limit[b])

    downsampling algo:
    we can use a lowpass filter (chebyshev type 1)
    decimation
    lets say min is 3431 and max is 3478 (47 points) and we want to downsample to 20 buckets

    Online from what I've read you have to upsample, LP filter, then downsample, to get a fraction of the upsample/downsample
    to resize the signal
    We can use polyphase arbitrary resampler!
    
    
    */