This repository contains a PIP package which is an OpenAI environment for
simulating a station-based car sharing system in which cars are rented.


## Installation

Install the [OpenAI gym](https://gym.openai.com/docs/).

Clone this repo: git clone https://github.com/ibrahim-elshar/Carsharing/gym_carsharing.git

Then install this package via

```
pip install -e .
```

## Usage

```
import gym
import adp_gym_carsharing

env = gym.make('AdpCarsharing-v0')
```



## The Environment

The environment consists of stations (by default 5 stations) from which cars are rented in accordance to a price-demand model with some noise.
After setting the price at each station the demand is observed and the destination stations are randomly assigned.
The time until arrival is proportional to the distance between the origin and destination stations.
The objective is to set the rental prices at each station during each period to maximize the total revenue.
An episode is 12 periods long.

