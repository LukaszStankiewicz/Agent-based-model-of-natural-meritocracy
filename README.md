# Agent based model of natural meritocracy

This is a model used in my paper: 

This README is an outline of the model, more information can be found in the code comments and the paper. 



## Purpose of the model

This is a simple model, created to see how meritocracy would work in a natural setting. It is based on the "farmers republic" idea, prevalent in USA in the XIX century. The model consists of agents trying to set up a farm. Agents are positioned randomly and can move if placed too near other agent. When they find a suitable place they start farming, each time step the farm grows proportionally to the merit parameter of the agent. When the farm touches any other farm the growth stops and the agent deactivates. 


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
In agent based modelling, looking at the model working in real-time can help in analysis. That being said the model here is rather boring to observe (farms grow then stop growing), but having a look might still be helpful to get a feel about its inner workings. 

Animated models are used for presentation rather then analysis (altought the model outcome can be saved). In order to initiate a model, you have to use command:

> sim = Simulator_simple_git()

And then launch a graphical simulation using:

> sim.start_simulation()

In order to save the outcome, use:

> sim.save_model_run()

#### Headless:
This allows to run scenarios with the same parameters many times, saving each outcome. 

In order to initiate a headless model, you have to use a function:

> save_multi_sims(iterations=10)

The model will be run number of times defined by the 'iteration' argument and the final outcomes will be saved automatically, with file name that encodes all the parameters of the model. 

In order to load the saved outcomes as a numpy array, use function:

> agents, text = load_multiple()

the load_multiple() function returns information for many runs of the model (agents) and the description of the parameters of the model (text). The format of the text is described at the end of the next section.  


## Model and function parameters

There are many variables that define the way model is run, they can be changed, when the animated or headless model is initialized.

The variables are:

#### seed_population
How many agents are created and take part in the model run. Default: 1000.

#### agent_speed
How far does the agent travel when it cannot find a free place for farming. Agents travel by a random value in range -speed / + speed simulatenously on x and y axes. Each point of speed moves an agent a percent of the model plane size. Default: 5.

#### plane_size
How big, in arbitrary units is the square plane on which agents work. Default: 1000. 

#### agent_contact_range
How far should another agents be from the active agent for the farming to start. Unit is the model plane size. Default: 0.001.

#### contact_range_falloff
If higher then zero, agent_contact_range falls by contact_range_falloff percent during each time step. If the agents are too picky about their farming site, their expectations will fall each round until they can find a proper place. Default: 0.

#### mean_merit
The mean value of agents' merit variable. The merit variable has a normal distribution based on this and next parameter. Default: 1.

#### merit_deviation
Standard deviation of the merit variable. Default: 0.2.

#### model_type
Takes values of 0 or 1. Determines what kind of capital accumulation takes place when agents work. 

Value of 0 means additive accumulation, wealth grows by merit parameter * 2 during each time step . 

Value of 1 is expotential accumulation. The capital The capital grows by merit % each time step.

Default: 1.

#### iterations_per_frame
How many time steps are simulated each time the agents are updated. 

It should take small values for animation mode, where it defines how many time steps ahead each new frame jumps (Default: 1 when using start_simulation() function). 

In headless mode it defines the limit of time steps for each simulation - and it should be large enough for all the agents to reach the end state (Default: 25000 when using save_multi_sims() function).

#### wall_stop
Should the walls stop the growth of the agents. 0 is no, 1 is yes. Default: 0. 


When loading saved runs of the model, the array with data will be accompanied by a string of text describing the model parameters. For instance:

multi_sim_iter_25000_spd_5_contr_0.001_contfall_0_contm_1000_mean_1_sd_0.2_modt_1_walls_False_ppl_1000__model_run_1

Is a model with 25000 iterations per frame, speed of 5, contact range of 0.001, contact_range_falloff of 0, mean merit of 1, merit deviation of 0.2, model type 1, wall_stop is set to False, agent population is 1000 - and it was saved as the first model run for the parameters stated. 


## Model outcome format
The model works on a numpy array containing agent information. The array is a table containing all agent information, with rows representing agents and columns their parameters. When the model is saved, the final version of this array is saved as a model outcome. Before saving some additional columns, calculated from the agent information are added to the table.

When the models are loaded en masse, the array is 3-dimensional as the tables representing each individual run are stacked. In this format 1st dimension are the model runs, 2nd are the agents, 3rd are their parameters. 

A single run can be represented as a pandas table by typing:

> show_new_agents(agents)

With 'agents' variable being a numpy array containing information about a model run. 

<img width="1218" height="385" alt="tabble" src="https://github.com/user-attachments/assets/2789f3db-b35d-458b-81e6-f8bc2d80b233" style="width:50%; height:auto;">>

Figure 2. The pandas table with agent information.

This gives us a table like this:

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







