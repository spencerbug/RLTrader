# DPT: Deep Population Trader

Creating a trading strategy using reinforcement learning, using multiple agents competing on a simulated exchange

This trading project uses a simulated exchange competing with other AI Agents, and getting historical Limit Order Book (LOB) data as a gym environment

## Plan

Each trading bot unit will consist of a portfolio manager (PM) and a 5 RL traders representing 5 stocks, to form a trading unit TU

* The portfolio managers job is to allocate resources among the 5 stocks.
* The 5 traders will place orders on the exchange directly

Both the the PM and traders are reinforcement learning algorithms and will use sharpe ratio based rewards.

There may be hundreds or thousands of TUs trading on the exchange and training, simultaneously, so we need to take that into account in the design, and the system must be scalable. To that end, we will build a scalable solution in google cloud possibly using the following services: kubernetes, pubsub, cloud functions and redis.

Hyperparameters will be calculated using population based gradient descent. The observations haven't been decided on yet, but it may involve an embedding layer as well. We can also use population data to eliminate unneeded observation spaces.

Traders will be given varying levels of market share, to account for different levels of control over the market

# Requirements

* Java 1.8
* Python 3.8
* mvn

# Building
`cd exchange-core`
`mvn install`
`cd exchange-gateway-rest`
`mvn install`