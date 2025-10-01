# Agent based model of natural meritocracy



This is a model used in my paper: 

This README is an outline of the model, more information can be found in the code comments and the paper. 

## Purpose of the model

This is a simple model, created to see how meritocracy would work in a natural setting. It is based on the "farmers republic" idea, prevalent in USA in the XIX century.


## About the model

The model was written in Python, using numpy library for calculations and numba to speed up iteration. It was originally based on an idea presented here: https://lrdegeest.github.io/blog/faster-abms-python.

There are two versions of the model: 


**Numba + numpy**: the default version, much faster but with a more awkward code. 

**Pure numpy**: needs only several simple libraries to work but is much slower.


Python code is generally almost-human readable but very slow. Numpy makes it faster at the cost of readability. Numba brings its own difficulties as many native numpy and Python commands can prevent it from working, so the code can get rather awkward and incosistent. I provided lots of in-code comments and simple functions so the model would be easy to use. 


## How to use the model?
There are two basic ways of using the model:

#### Animated: 
In agent based modelling, looking at the model working in real-time can help in analysis. That being said the model here is rather boring to observe (farms grow then stop growing), but having a look might still be helpful to get a feel about its inner workings. 

Animated models are used for presentation rather then analysis (altought the model outcome can be saved). In order to initiate a model, you have to use command:

> sim = Simulator_simple_git()

And then launch a graphical simulation using:

> sim.start_simulation()

In order to save the outcome, use:

> sim.save_model_run()


#### Headless:
This allows to run scenarios with the same parameters many times, saving the outcome of the model. 

In order to initiate a headless model, you have to use a function:

> save_multi_sims()

The model will be run and the final outcome saved automatically, with file name that encodes all the parameters of the model. 

In order to load the saved outcomes as a numpy array, use function:

> agents, text = load_multiple()

the load_multiple() function returns information for many runs of the model (agents) and the description of the parameters of the model (text). 


## Model and function parameters

There are many variables that define the way model is run, they can be changed, when the animated or headless model is initialized.

The variables are:

#### seed_population
How many agents are created and take part in the model run. Default: 1000.

#### agent_speed
How far does the agent travel when it cannot find a place free for farming. Agents travel by a random value in range -speed / + speed simulatenously on x and y axes. Unit is a percent of the model plane size. Default: 5.

#### plane_size
How big, in arbitrary units is the square plane on which agents work. Default: 1000. 

#### agent_contact_range
How far should one agent be from another in order to decide that it is not to close to farm. Unit is the model plane size. Default: 0.001.

#### contact_range_falloff
If higher then zero agent_contact_range falls each time step by contact_range_falloff percent. Default: 0.

#### mean_merit
The mean value of agents merit variable. The merit variable has a normal distribution based on this and next parameter. Default: 1.

#### merit_deviation
Standard deviation of the merit variable. Default: 0.2.

#### model_type
Takes values of 0 or 1. Determines what kind of capital accumulation takes place when agents work. Value of 0 means additive accumulation, each time step capital grows by merit parameter * 2. Value of 1 is expotential accumulation. Each time step the capital is multiplied by 1 + merit parameter divided by 100. Default: 1.

#### iterations_per_frame
How many time steps are simulated each time the agents are updated. It should take small values for animation, where it defines how many time steps ahead each new frame jumps (Default: 1 when using sim.start_simulation() function). In headless mode it defines the limit of time steps for each simulation - and it should be large enough for all agents to reach the end state (Default: 25000 when using save_multi_sims() function).

#### wall_stop
Should the walls stop the growth of the agents. 0 means no, 1 means yes. Default: 0. 


## Model outcome format
The model works on a numpy array containing agent information. The array is a table containing all agent information, with rows representing agents and columns their parameters. When the model is saved, the final version of the array is saved on the disk. Before saving some additional columns, calculated from the agent information are added to the table.

When the models are loaded en masse, the array is 3-dimensional: the tables representing each individual run are stacked. In this format 1st dimension are the model runs, s2nd are the agents, 3rd are their parameters. 

A single run can be represented as a pandas table by typing:

> show_new_agents(agents)

With 'agents' variable being a numpy array containing information about a model run. 

The format of the numpy array is at follows:

0: agent's id

1, 2: agent's x and y position

3, 4: agent's speed in a time step

5: agent's size

6: agent's merit

7: accumulated capital

8: agent's state (0 for looking for a place to farm, 1: farming, 2: inactive)

9: total production time

10, 11: agent's x and y starting position

12: distance to the nearest other agent

13: merit of nearest agent

14: is the agent the nearest agent for its nearest agent?

15: agent's distance to nearest wall




## Functions and other stuff
The code comments contain information about the function parameters and ways to change variables of both animated ad headless models. 







