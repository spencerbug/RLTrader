//limit order book
// reference: https://gist.github.com/halfelf/db1ae032dc34278968f8bf31ee999a25
// https://github.com/Kautenja/limit-order-book/blob/master/notes/lob.md

#ifndef _LOB_H
#define _LOB_H

#include <unordered_map>
#include <vector>

typedef enum {SELLSIDE, BUYSIDE} Side;
enum Color {RED, BLACK};

struct Order {
    long orderId; //keyed
    Side side;
    long qty;
    long limit;
    long entryTime;
    long eventTime;
    Order *prevOrder;
    Order *nextOrder;
    Limit *parentLimit; 
};

struct Limit {
    long limitPrice; //keyed
    long size;
    long totalVolume;
    Limit *parent;
    Limit *leftChild;
    Limit *rightChild;
    bool color;
    Order *headOrder;
    Order *tailOrder;
    Limit(long limitPrice){
        this->color=RED;
        this->leftChild=nullptr;
        this->rightChild=nullptr;
        this->headOrder=nullptr;
        this->tailOrder=nullptr;
        this->limitPrice=limitPrice;
        this->size=0;
        this->totalVolume=0;
    }
};

//RBTree and hashmap combo for limits
class LimitTree{
    private:
        Limit *root;
        unordered_map<long, Limit*> limitmap;
    protected:
        void rotateLeft(Limit *&, Limit *&);
        void rotateRight(Limit *&, Limit *&);
        void fixViolation(Limit *&, Limit *&);
    public:
        LimitTree() { root=nullptr; }
        void insert(long Limit);
        void remove(long Limit);
        vector<vector<long>> inorder();//traverses the tree and gets vector of {price,volume} in order
};

class Book {
    LimitTree *buyTree;
    LimitTree *sellTree;
    Limit *lowestSell;
    Limit *highestBuy;
    
    //places order at the end of a list of orders to be execute at a limit price
    //O(log M) for first order to add at a limit, O(1) for others. Using a binary tree and a hash(of limit) table for that
    long add(Side side, long qty, long limit, long time);

    //removes an arbitrary order
    void cancel(long orderId); // O(1) hash lookup

    //removes an order from inside the book. "inside" is oldest buy at highest price and oldest sell at lowest price
    void execute(); //performs optimal execution between a buy and sell side

    // gets the volume at a given limit
    long getVolumeAtLimit(Side side, long limit); //O(1)
    
    // gets the optimal order for a given limit
    Order *getBestBidOrOffer(long limit, Side side); // O(1)

};

#endif