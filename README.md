# Agent based model of natural meritocracy

This is a model used in my paper: https://zenodo.org/records/17290173

This README is an outline of the model, more information can be found in the code comments and the paper. 



## Purpose of the model

This is a simple model, created to see how meritocracy would work in a natural setting. It is based on the "farmers republic" idea, prevalent in USA in the XIX century. The model consists of agents trying to set up a farm. Agents are positioned randomly ona  square plan. They can move if placed too near other agents or farms. When they find a suitable place they start farming and each time step the farm grows proportionally to the merit parameter of the agent. When the farm touches any other farm the growth stops and the agent deactivates. 


<img width="1225" height="1225" alt="Figure 1 (1)" src="https://github.com/user-attachments/assets/4f592e57-6ce5-401a-acf6-4ef86a47244f" style="width:33%; height:auto;">>

Figure 1. Simulation mid-run. 


## About the model

The model was written in Python, using numpy library for calculations and numba to speed up iteration. It was originally based on an idea presented here: https://lrdegeest.github.io/blog/faster-abms-python.

There are two versions of the model: 


**Numba + numpy**: the default version, much faster. 

**Pure numpy**: needs only several simple libraries to work but is much slower.


Python code is generally almost-human readable but very slow. Numpy makes it faster at the cost of readability. Numba brings its own difficulties as many native numpy and Python commands can prevent it from working, so the code can get rather awkward and incosistent (but it's very, very fast). 

I provided lots of in-code comments and simple functions so the model is easy to use. 


## How to use the model?
There are two basic ways of using the model:

#### Animated: 
In agent based modelling, looking at the model working in real-time can help the analysis. That being said the model here is rather boring to observe (farms grow then stop growing), but having a look might still be helpful to get a feel about its inner workings. 

In order to start the animated mode, you have to initialize the main class:

> sim = Simulator_simple_git()

And then launch animated simulation using:

> sim.start_simulation()

In order to save the outcome, use:

> sim.save_model_run()

#### Headless:
This mode runs a simulation many times, saving each outcome. 

In order to initiate a headless model, you have to use a function:

> save_multi_sims(iterations=10)

The model will be run number of times defined by the 'iteration' argument and the final outcomes will be saved automatically, with file name that encodes all the parameters of the model. 

In order to load the saved outcomes as a numpy array, use function:

> agents, text = load_multiple()

the load_multiple() function returns information for many runs of the model (agents) and the description of the parameters of the model (text). The format of the description is outlined at the end of the next section.  


## Model and function parameters

There are many variables that define the way model is run, they can be changed, when the animated or headless model is initialized. Their default values are based on the model configuration described in the paper linked to at the beggining of the README.

The variables are:

#### seed_population
How many agents are active in a single simulation run. Default: 1000.

#### agent_speed
How far does an agent travel when it cannot find a suitable place for farming. Agents travel by a random value in range -speed / + speed simulatenously on x and y axes. Each point of speed moves an agent 1% of the model plane size. Default: 5.

#### plane_size
What is the width and height, in arbitrary units, of the square plane on which agents are placed. Default: 1000. 

#### agent_contact_range
How far should another agents be from the active agent for the farming to start. Unit is the model plane size. Default: 0.001.

#### contact_range_falloff
If higher then zero, agent_contact_range falls by contact_range_falloff percent during each time step. This mean that if the agents are too picky about their farming site, their expectations will fall each round until they can find a proper place. Default: 0.

#### mean_merit, merit_deviation
Two parameters defining the distribution of agents' merit. Merit is normally distributed. Defaults: 1, 0.2.

#### model_type
Takes values of 0 or 1. Determines what kind of capital accumulation takes place when agents work. 

Value of 0 means additive accumulation, wealth grows by merit parameter * 2 during each time step . 

Value of 1 is expotential accumulation. Wealth grows by merit % each time step.

Default: 1.

#### iterations_per_frame
How many time steps are simulated each time the agents are updated. 

It should take small values for animation mode, where it defines how many time steps ahead each frame jumps (Default: 1 when using start_simulation() method). 

In headless mode it defines the limit of time steps for each simulation - and it should be large enough for all the agents to reach their end states (Default: 25000 when using save_multi_sims() function).

#### wall_stop
Tells the sim if the walls should stop the growth of agents' farms. 0 is no, 1 is yes. Default: 0. 

<br/><br/>
When loading saved runs of the model, the array with data will be accompanied by a string describing the model parameters. For instance:

multi_sim_iter_25000_spd_5_cont_0.001_contfall_0_plane_1000_mean_1_sd_0.2_model_1_walls_False_ppl_1000__model_run_1

Is a model with 25000 iterations per frame, speed of 5, contact range of 0.001, contact_range_falloff of 0, mean merit of 1, merit deviation of 0.2, model type 1, wall_stop set to False, agent population of 1000 - and it was saved as the first model run for the parameters stated. 


## Model outcome format
The model works by modyfing a numpy array containing agent information. The array is a table with rows representing agents and columns - their parameters. When the model is saved, the final version of this array is saved as a model outcome. Before saving, additional columns calculated from the agent information are added to the table.

When the saved model runs are loaded en masse, the array is 3-dimensional as the tables representing each individual run are stacked. In this format 1st dimension are the model runs, 2nd are the agents, 3rd are their parameters. 

A single run can be represented as a pandas table by typing:

> show_new_agents(agents)

With 'agents' variable being a numpy array containing information about a model run. 

This gives us a table like this:

<img width="1218" height="385" alt="tabble" src="https://github.com/user-attachments/assets/2789f3db-b35d-458b-81e6-f8bc2d80b233" style="width:50%; height:auto;">>

Figure 2. The pandas table with agent information.


The format of the numpy array is at follows:

0     : agent's id

1, 2  : agent's x and y position

3, 4  : agent's x and y speed during the last time step

5     : agent's size 

6     : agent's merit

7     : accumulated wealth

8     : agent's state (0 for looking for a place to farm, 1: farming, 2: inactive)

9     : total production time

10, 11: agent's x and y starting position

12    : distance to the nearest other agent

13    : merit of nearest agent

14    : is the agent the nearest agent for its nearest agent?

15    : agent's distance to the nearest wall




## Functions and other stuff
The code comments contain information about the function parameters and ways to change variables of both animated and headless modes. 







