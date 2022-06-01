using namespace std;



#include "lob.h"

//https://www.geeksforgeeks.org/c-program-red-black-tree-insertion/

void inorderHelper(Limit *root, vector<Limit *> &result){
    if (root==nullptr) return;
    inorderHelper(root->leftChild, result);
    result.emplace_back(root);
    inorderHelper(root->rightChild, result);
}

Limit *BSTInsert(Limit *root, Limit *pt){
    // if tree is empty, return a new tree
    if (root==nullptr) pt;

    // otherwise recur down the tree
    if (pt->limitPrice < root->limitPrice){
        root->leftChild = BSTInsert(root->leftChild, pt);
        root->leftChild->parent = root;
    } else if (pt->limitPrice > root->limitPrice){
        root->rightChild = BSTInsert(root->rightChild, pt);
        root->rightChild->parent = root;
    }

    return root;
}

void LimitTree::rotateLeft(Limit *&root, Limit *&pt){
    Limit *pt_right = pt->rightChild;
    pt->rightChild = pt_right->leftChild;
    if (pt->rightChild != nullptr)
        pt->rightChild->parent = pt;
    pt_right->parent = pt->parent;
    if (pt->parent == nullptr)
        root = pt_right;
    else if (pt == pt->parent->leftChild)
        pt->parent->leftChild = pt_right;
    else
        pt->parent->rightChild = pt_right;
    pt_right->leftChild = pt;
    pt->parent = pt_right;
}