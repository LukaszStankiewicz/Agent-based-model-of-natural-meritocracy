# Agent based model of natural meritocracy
---


This is a model used in my paper: 


## Purpose of the model
---
This is a simple model, created to see how meritocracy would work in a natural setting. It is based on the "farmers republic" idea, prevalent in USA in the XIX century.

## About the model
---
The model was written in Python, using numpy library for calculations and numba to speed up iteration. It was originally based on an idea presented here: https://lrdegeest.github.io/blog/faster-abms-python.
There are two versions of the model: 

**Numba + numpy**: the default version, much faster but with a more awkward code. 

**Pure numpy**: needs only several simple libraries to work but is much slower.

Python code is generally almost-human readable but very slow. Numpy makes it faster at the cost of readability. Numba brings its own difficulties as many native numpy and Python commands can prevent it from working, so the code can get rather awkward and incosistent. I provided lots of in-code comments and simple functions so the model would be easy to use. 

## How to use the model?
There are two basic ways of using the model:

**Animated**: in agent based modelling, looking at the model working in real-time can be very important for analysis. That being said the model here is rather boring to observe (farms grow then stop growing), but it might still be useful in order to get a feel about its inner workings. 

Animated models are used for presentation rather then analysis (altought the model outcome can be saved). In order to initialie a model, you have to use commands:

> sim = Simulator_simple_git()

This launches the main class of the model.

> sim.start_simulation()

This launches the graphical simulation.








