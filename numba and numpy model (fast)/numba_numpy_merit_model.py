import numpy as np
import numba
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import os
from os import listdir
from os.path import isfile, join
import black


# CLASS
class Simulator_simple_git:
    def __init__(
        self,
        params_dict={
            "seed_population": 1000,
            "agent_speed": 5,
            "plane_size": 1000,
            "agent_contact_range": 0.001,
            "contact_range_falloff": 0,
            "mean_merit": 1,
            "merit_deviation": 0.2,
            "model_type": 1,
            "iterations_per_frame": 1,
            "wall_stop": False,
        },
    ):
        """
        This is the main class of the model. It contains methods that create agents and run the model in animated and headless modes.
        Model parameters are defined by this class' arguments - particularily the params_dict, a dictionary containing all the variables
        of the model. The default values for all the variables are based on the main scenario described in the paper based on this model.

        :param params_dict             (python dictionary) : dictionary containing model parameters.
        :param params_dict["seed_population"]        (int) : how many agents are created and take part in the model run.
        :param params_dict["agent_speed"]          (float) : How far does the agent travel when it cannot find a free place for farming.
                                                             Agents travel by a random value in range -speed to + speed simulatenously
                                                             on x and y axes. Each point of speed moves an agent a percent of the model
                                                             plane size
        :param params_dict["plane_size"]             (int) : How big, in arbitrary units is the square plane on which agents work.
        :param params_dict["agent_contact_range"]  (float) : how far should another agents be from the active agent for the farming to
                                                             start. Unit is the model plane size.
        :param params_dict["contact_range_falloff"](float) : If higher then zero, agent_contact_range falls by contact_range_falloff
                                                             percent during each time step. If the agents are too picky about their
                                                             farming site, their expectations will fall each round until they can find
                                                             a proper place.
        :param params_dict["mean_merit"]           (float) : The mean value of agents' merit variable. The merit variable has a normal
                                                             distribution based on this and next parameter.
        :param params_dict["merit_deviation"]      (float) : Standard deviation of the merit variable.
        :param params_dict["model_type"]            (bool) : Takes values of 0 or 1. Determines what kind of capital accumulation takes
                                                             place when agents work.
                                                             Value of 0 means additive accumulation, wealth grows by merit parameter * 2
                                                             during each time step .
                                                             Value of 1 is expotential accumulation. The capital grows by merit % each
                                                             time step.
        :param params_dict["iterations_per_frame"]  (bool) : How many time steps are simulated each time the agents are updated.
                                                             It should take small values for animation mode, where it defines how many time steps
                                                             ahead each new frame jumps. Default is 1 when using start_simulation() function.
                                                             In headless mode it defines the limit of time steps for each simulation - and it
                                                             should be large enough for all the agents to reach the end state. Default is 25000
                                                             when using save_multi_sims() function.
        :param params_dict["wall_stop"]             (bool) : Should the walls stop the growth of the agents. 0 is no, 1 is yes.
        """

        # Sets model variables.
        self.seed_population = params_dict["seed_population"]
        self.agent_speed = params_dict["agent_speed"]
        self.plane_size = params_dict["plane_size"]
        self.agent_contact_range = params_dict["agent_contact_range"]
        self.contact_range_falloff = params_dict["contact_range_falloff"]
        self.mean_merit = params_dict["mean_merit"]
        self.merit_deviation = params_dict["merit_deviation"]
        self.model_type = params_dict["model_type"]
        self.iterations_per_frame = params_dict["iterations_per_frame"]
        self.wall_stop = params_dict["wall_stop"]

        # Creates agents.
        self.agents = self.setup_merit_agents()

    # Class functions
    def setup_merit_agents(self):
        """
        Generates agents for simulation. Each agent is defined by its parameters, described below.

        Agent parameters:
                0      id                        an id number, different for each agent in a simulation
                1,2    x/y position              position on x and y axes of the simulation plane
                3,4    x/y momentum              speed of agents, randomized each time agent moves
                5      agent_size                agent size, defined by its accumulated capital
                6      agent_merit               agent merit parameter
                7      agent_accumulation        how much wealth agent accumulated during the run
                8      agent state               states are: 0: looking for production site, 1: producing, 2: inactive
                9      total production time     how much time did the agent spent producing
                10,11  starting x/y              starting x and y positions of an agent

        :returns agents (numpy array): returns numpy array containing agent information.
        """

        # Generates numpy array for agent parameters.
        agents = np.zeros((self.seed_population, 12), dtype=float)

        # Fills the array with parameters.
        agents[:, 0] = np.arange(self.seed_population)
        agents[:, 1] = np.random.uniform(0, 1, self.seed_population)
        agents[:, 2] = np.random.uniform(0, 1, self.seed_population)
        agents[:, 3] = np.zeros(self.seed_population)
        agents[:, 4] = np.zeros(self.seed_population)
        agents[:, 5] = np.ones(self.seed_population)
        agents[:, 6] = np.random.normal(
            self.mean_merit, self.merit_deviation, self.seed_population
        )
        agents[:, 7] = np.ones(self.seed_population)
        agents[:, 8] = np.zeros(self.seed_population)
        agents[:, 9] = np.zeros(self.seed_population)
        agents[:, 10] = agents[:, 1]
        agents[:, 11] = agents[:, 2]

        # Sets agents' size based on starting capital.
        agents[:, 5] = np.sqrt(agents[:, 7] / 3.14)

        # Automatically puts agents that will not move during the simulation in production mode.
        if self.agent_speed == 0:
            agents[:, 8] = 1

        return agents

    def setup_plot(self):
        """
        Generates the plot for the first time, placing agents on it. This plot is used for animations and it does not
        scale agents sizes properly. To get a picture of agents in real scale use real_scale_plot below.
        """
        self.fig = plt.figure(figsize=(7, 7))
        self.ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
        self.scatter = self.ax.scatter(
            self.agents[:, 1],
            self.agents[:, 2],
            s=self.agents[:, 5] ** 2
            / 1.75,  # this calculation allows agents' size on the plot to be approximately their "real" size
            alpha=0.5,
            color="green",
        )

    def real_scale_plot(self, agents, scale=1000, show_colors=False):
        """
        This method is not used during the simulation but it can be called to generate a graphics of agents in real scale.

        :param agents     (numpy array) : This method needs an array with agent data from a single simulation run.
        :param scale      (int)         : Scale parameter needs to be provided, it defaults to 1000 and should be the same
                                          as plane_size used for the simulation.
        :param show_color (bool)        : Defaults to False, if True agents' colors differ based on their size.
        :returns scatter (matplotlib graphics) : Plot of agents in real scale.
        """
        # Generates x and y.
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)

        # Calculates agents' size on the plot.
        s = np.sqrt(agents[:, 7] / np.pi) / scale

        # Generates the plot.
        colors = [cm.binary(color) for color in agents[:, 6] / np.max(agents[:, 6])]
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = plt.gca()

        # Adds circles representing agents to the plot.
        if show_colors == True:
            for a, b, size, color in zip(agents[:, 1], agents[:, 2], s, colors):
                circle = plt.Circle(
                    (a, b),
                    size,
                    color=color,
                    edgecolor="black",
                    alpha=0.7,
                    linewidth=0.5,
                )
                self.ax.add_artist(circle)
        else:
            for a, b, size in zip(agents[:, 1], agents[:, 2], s):
                circle = plt.Circle(
                    (a, b), size, edgecolor="black", alpha=0.7, linewidth=0.5
                )
                self.ax.add_artist(circle)

        # Sets ax limits and aspect ratio.
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect(1.0)

        # Plots the plot.
        self.scatter = plt.scatter(
            x, y, s=0, cmap="binary", facecolors="none", alpha=0.1
        )

        return self.scatter

    def update_plot(self, i):
        """
        Method that provides frames for the animated mode of the simulation.

        :param i         (int)                : frame number, provided by the animation function.
        :returns scatter (matplolib graphics) : a frame for the animation function.
        """
        # Clears the plot.
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)

        # Updates agents.
        self.agents = self.update_agents(
            self.agents,
            self.iterations_per_frame,
            self.agent_speed,
            self.agent_contact_range,
            self.contact_range_falloff,
            self.plane_size,
            i,
            self.model_type,
            self.wall_stop,
        )

        # Updates the plot using new agent infromation.
        self.scatter = self.ax.scatter(
            self.agents[:, 1],
            self.agents[:, 2],
            s=self.agents[:, 5] ** 2 / 1.75,
            alpha=0.5,
            color="green",
        )
        self.scatter = self.ax.scatter(
            self.agents[:, 1][self.agents[:, 8] == 2],
            self.agents[:, 2][self.agents[:, 8] == 2],
            s=self.agents[:, 5][self.agents[:, 8] == 2] ** 2 / 1.75,
            alpha=1,
            color="black",
        )

        # Stops the animation if all agents reached an end state.
        if len(self.agents[self.agents[:, 8] == 2]) == len(self.agents):
            self.ani.event_source.stop()
            self.annot = self.ax.annotate(
                f"ALL AGENTS INACTIVE", (0.1, 0.9), xycoords="figure fraction"
            )
            return self.scatter, self.annot

        return self.scatter

    @staticmethod
    @numba.njit
    def update_agents(
        agents,
        iterations_per_frame,
        agent_speed,
        agent_contact_range,
        contact_range_falloff,
        plane_size,
        frame,
        model_type,
        wall_stop,
    ):
        """
        The main method running the simulation. It can be called by update_plot method if it is used in animated mode, or
        by the save_multi_sims function if it is called to generate and save simulation runs in a 'headless' (non-graphical) mode.
        This is a numba metod - numba is a library that makes the calculations much faster (and allows to run and save many simulations
        quickly) - but it also necessisates a rather awkward syntax, as it is not compatible with many python and numpy commands.
        First difference between this and any other method is that all the variables must be provided to the method as arguments
        as it cannot import the in-class variables.
        This is done automatically in animated mode, and must be done manually (or by a functions such as save_multi_sims) when this
        mwthod is called from outside the class.
        If the function is called from outside the variables provided manually override any variables that were set in the frames_dict
        while initializing the Simulator_simple_git class.

        :params agents         (numpy arr)   : Numpy array containing agent data.
        :params iterations_per_frame (int)   : If defined while initiating the function and using it in animated mode this variable says
                                               how many time steps should there be in a single frame of the animation (default is 1).
                                               If called in headless mode, it defines how many time steps will there be in the whole
                                               simulation. Setting it too low might end the simulation before all the agents reach an
                                               end state. Default for the save_multi_sims function is 25,000 but it can be set arbitrarily
                                               high as the run ends automatically as soon as all agents reach the end state.
        :params agent_speed, agent_contact_range, contact_range_falloff, plane_size, model_type, wall_stop: are the general variables
                                               that define any simulation, see the description of the main class.
        :param frame                 (int)   : Set automatically in animated mode. Set to 1 in headless mode.
        :returns agents      (numpy array)   : numpy array containing agent data.
        """

        # Agents are updated x times, accoring to the value of iterations_per_frame argument.
        for x in range(iterations_per_frame):

            # Updates contact_range if contact range is set to diminish each time step (by a contact_range_falloff being more than 0).
            if agent_contact_range > 0:
                if contact_range_falloff > 0:
                    agent_contact_range = agent_contact_range - (
                        x * 0.01 * contact_range_falloff * agent_contact_range
                    )

            # Resets agents' momentum.
            agents[:, 3] = 0
            agents[:, 4] = 0

            # Determines who is near whom.
            # Creates array for listing all agents that are near other agents. Value of 1 in this array means that agents are near each other.
            near_arr_xy = np.zeros((len(agents), len(agents)))

            # Create array for listing agents that are close to walls.
            wall_arr = np.zeros((len(agents), 4))

            # Finds active agents in range of other agents, and agents that are near to walls.
            for z in range(len(agents)):

                # Finds agents that are near other agents.
                distance = np.sqrt(
                    (np.abs(agents[:, 1] - agents[z, 1])) ** 2
                    + (np.abs(agents[:, 2] - agents[z, 2])) ** 2
                ) < (agent_contact_range + ((agents[:, 5] + agents[z, 5]) / plane_size))
                near_arr_xy[z] = distance
                near_arr_xy[z, z] = False

                # Find agents that are near walls.
                wall_arr[z, 0] = agents[z, 1] < (
                    agent_contact_range + agents[z, 5] / plane_size
                )
                wall_arr[z, 1] = 1 - agents[z, 1] < (
                    agent_contact_range + agents[z, 5] / plane_size
                )
                wall_arr[z, 2] = agents[z, 2] < (
                    agent_contact_range + agents[z, 5] / plane_size
                )
                wall_arr[z, 3] = 1 - agents[z, 2] < (
                    agent_contact_range + agents[z, 5] / plane_size
                )

            # Below is the state machine that decides what agents do.

            # STATE 0: Looking for production site.

            # First checks if agents are in an empty place and can start producing. If they are it changes their state to 1 (production).
            for x in range(len(agents)):
                if agents[x, 8] == 0:
                    if np.any(near_arr_xy[x, :]) == False:
                        agents[x, 8] = 1

            # Agents that cannot start production, have momentum parameters set. They will be moved later.
            if len(agents[agents[:, 8] == 0]) > 0:
                agents[:, 3][agents[:, 8] == 0] = np.random.uniform(
                    -0.01, 0.01, len(agents[agents[:, 8] == 0])
                )
                agents[:, 4][agents[:, 8] == 0] = np.random.uniform(
                    -0.01, 0.01, len(agents[agents[:, 8] == 0])
                )

            # STATE 1: Production.
            for y in range(len(agents)):
                # Selects all producers.
                if agents[y, 8] == 1:

                    # Stops production (sets their state to 2: end state) for agents with merit less than zero.
                    if agents[y, 6] < 0:
                        agents[y, 8] = 2

                    # Stops production for agents whose farms touch other farms.
                    for x in range(len(agents)):
                        if (agents[x, 8] == 1) or (agents[x, 8] == 2):
                            if near_arr_xy[x, y] == True:
                                agents[y, 8] = 2
                                break

                    # If parameter wall_stop is True, stops production for agents that touch the walls.
                    if wall_stop == True:
                        if np.any(wall_arr[y, :]) == True:
                            agents[y, 8] = 2
                            break

                    # Begins production for agents whose state is still 1 (production).
                    if agents[y, 8] == 1:

                        # Production function for additive model of production.
                        if model_type == 0:
                            agents[y, 7] += agents[y, 6] * 2

                        # Production function for exponential model of production.
                        elif model_type == 1:
                            agents[y, 7] *= 1 + (agents[y, 6] / 100)

                        # Updates agent's size and total production time parameters.
                        agents[y, 5] = np.sqrt(agents[y, 7] / 3.14)
                        agents[y, 9] += 1

            # Moves agents that had their momentum parameter set above 0.
            agents[:, 1] = agents[:, 1] + (agents[:, 3] * agent_speed)
            agents[:, 2] = agents[:, 2] + (agents[:, 4] * agent_speed)

            # Keeps agents in the plane.
            agents[:, 1][agents[:, 1] < 0] = 0
            agents[:, 2][agents[:, 2] < 0] = 0
            agents[:, 1][agents[:, 1] > 1] = 1
            agents[:, 2][agents[:, 2] > 1] = 1

            # If all agents are in state 2 (end state) it ends the model run.
            if np.all(agents[:, 8] == 2):
                return agents

        return agents

    def start_simulation(self):
        """
        This method generates a plot, checks if agents need refreshing and starts an animated model run.
        """
        self.setup_plot()

        if np.any(self.agents[:, 8] == 2):
            self.agents = self.setup_merit_agents()

        self.ani = FuncAnimation(
            fig=self.fig,
            func=self.update_plot,
            interval=0.1,
            blit=True,
            cache_frame_data=False,
        )

    def calculate_and_add_columns(self, agents):
        """
        This method calculates some parameters from an array containing an information on a model run and adds them to the array. The information
        calculated for each agent are: distance to nearest agent, merit of nearest agent, does the nearest agent also has the agent as the nearest
        agent (0 is no, 1 is yes), distance to the nearest wall.

        :param agents       (numpy array) : array with agents information.
        :returns agents_new (numpy array) : the same array but with additional columns added.
        """
        # Creates empty arrays for the calculated data.
        distance_array = np.zeros(len(agents))
        near_merit_array = np.zeros(len(agents))
        close_array = np.zeros(len(agents))
        near_index_matrix = np.zeros(len(agents))

        # Calculates distances to other agents, chooses the nearest one and gets it merit. Puts the data into empty arrays.
        for x in range(len(agents)):
            dist_x = np.abs(agents[:, 1] - agents[x, 1])
            dist_y = np.abs(agents[:, 2] - agents[x, 2])
            dist_z = np.sqrt(dist_x**2 + dist_y**2) * self.plane_size
            dist_z[x] = (
                self.plane_size * 2
            )  # this sets distance to self at (much) more then 0 so that agents would not be considered nearest to themselves.
            k_near = np.argmin(dist_z)

            distance_array[x] = dist_z[k_near]
            near_merit_array[x] = agents[k_near, 6]
            near_index_matrix[x] = agents[k_near, 0]

        # Checks if nearest agents also have tha active agent as their nearest.
        for y in range(len(near_index_matrix)):
            if near_index_matrix[int(near_index_matrix[y])] == y:
                close_array[y] = 1

        # Checks distance of agents to a wall.
        close_to_walls_array = (
            np.min(
                np.array(
                    [agents[:, 1], 1 - agents[:, 1], agents[:, 2], 1 - agents[:, 2]]
                ),
                axis=0,
            )
            * 1000
        )

        # Adds the arrays with calculated data as columns to the main agents array.
        agents_new = np.array(
            ([agents[:, x] if x < 12 else distance_array for x in range(13)])
        )
        agents_new = agents_new.T
        agents_new = np.array(
            ([agents_new[:, x] if x < 13 else near_merit_array for x in range(14)])
        )
        agents_new = agents_new.T
        agents_new = np.array(
            ([agents_new[:, x] if x < 14 else close_array for x in range(15)])
        )
        agents_new = agents_new.T
        agents_new = np.array(
            ([agents_new[:, x] if x < 15 else close_to_walls_array for x in range(16)])
        )
        agents_new = agents_new.T

        return agents_new

    def save_model_run(self, mypath="", root_file_name="multi_sim_"):
        """
        This rather long method saves the run from animated mode to the disk. It must do a lot of additonal work
        because each run has to be properly named and numbered.

        :param mypath         (string) : this is the place to set the path to a folder for the saved simulations.
                                         If it is left empty, it defaults to 'SAVED_SIMS': a subfolder of a folder
                                         where the simulation files are placed.
        :param root_file_name (string) : sets the beggining of a file name for all the saved simulations.

        """
        # If mypath argument is empty, creates the 'SAVED_SIMS' subfolder.
        if mypath == "":
            os.makedirs(os.getcwd() + "/SAVED_SIMS", exist_ok=True)
            mypath = os.getcwd() + "/SAVED_SIMS"

        # Creates a file name that contains the information about model parameters.
        file_name_basic = (
            root_file_name
            + f"""iter_{self.iterations_per_frame}_
                                           spd_{self.agent_speed}_
                                           cont_{self.agent_contact_range}_
                                           contfall_{self.contact_range_falloff}_
                                           plane_{self.plane_size}_
                                           mean_{self.mean_merit}_
                                           sd_{self.merit_deviation}_
                                           model_{self.model_type}_
                                           walls_{self.wall_stop}_
                                           ppl_{self.seed_population}""".replace(
                "\n", ""
            ).replace(
                " ", ""
            )
        )

        file_name = file_name_basic + "__model_run_1"

        # Creates a list of all the data in the save folder.
        datafile_list = [f for f in listdir(mypath) if isfile(join(mypath, f))]

        # If in a saved folder there are simulations with the same name (which means: the same model parameters) and same numbering
        # then it finds the first free number for the saved file.
        while f"{file_name}.npy" in datafile_list:
            saved_file_number = int("".join(list(filter(str.isdigit, file_name[-4:]))))
            new_number = saved_file_number + 1
            file_name = file_name[:-4] + file_name[-4:].replace(
                str(saved_file_number), str(new_number)
            )

        # Establishes the final file name and path.
        saved_file_name = file_name
        final_path = mypath + "/" + saved_file_name

        # Prepares the array to be saved by sending it to a function that calculates additional prameters.
        agents_final = self.calculate_and_add_columns(self.agents)

        # Saves the array as a npy file.
        np.save(final_path, agents_final)
        print("Saved file as: ", saved_file_name)


def save_multi_sims(
    iterations=10,
    seed_population=1000,
    iterations_per_frame=25000,
    agent_speed=5,
    plane_size=1000,
    agent_contact_range=0.001,
    contact_range_falloff=0,
    mean_merit=1,
    merit_deviation=0.2,
    model_type=1,
    wall_stop=False,
    mypath="",
    root_file_name="multi_sim_",
):
    """
    This is the main function for the headless simulation mode. It runs the simulation x times, with the parameters of the model provided
    as the function arguments, and saves all the runs at a provided path or in a "SAVED_SIMS" folder it creates for the purpose.

    :param iterations           (int) : how many model runs should be saved.
    :param seed_population      (int) : how many agents in each simulation.
    :param iterations_per_frame (int) : maximum time steps in a model run. Should be set to 25,000 or more.
    :param agent_speed, plane_size, agent_contact_range, contact_range_falloff, mean_merit, merit_deviation, model_type, wall_stop: are
                                        the same general variables that define any simulation, see the description of the main class.

    :param mypath            (string) : this is the place to set the path to a folder for the saved simulations.
                                         If it is left empty, it defaults to 'SAVED_SIMS': a subfolder of a folder
                                         where the simulation files are placed.
    :param root_file_name    (string) : sets the beggining of a file name for all the saved simulations.
    """
    # If mypath argument is empty, creates the 'SAVED_SIMS' subfolder.
    if mypath == "":
        os.makedirs(os.getcwd() + "/SAVED_SIMS", exist_ok=True)
        mypath = os.getcwd() + "/SAVED_SIMS"

    # Creates a list of all the data in the save folder.
    datafile_list = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    # Initiates the main class (Simulator_simple_git) and provides it with parameters based on this function arguments.
    sim = Simulator_simple_git(
        params_dict={
            "iterations_per_frame": iterations_per_frame,
            "seed_population": seed_population,
            "agent_speed": agent_speed,
            "plane_size": plane_size,
            "agent_contact_range": agent_contact_range,
            "contact_range_falloff": contact_range_falloff,
            "mean_merit": mean_merit,
            "merit_deviation": merit_deviation,
            "model_type": model_type,
            "wall_stop": wall_stop,
        }
    )

    # Creates a file name that contains the information about model parameters.
    file_name_basic = (
        root_file_name
        + f"""iter_{iterations_per_frame}_
                                           spd_{agent_speed}_
                                           cont_{agent_contact_range}_
                                           contfall_{contact_range_falloff}_
                                           plane_{plane_size}_
                                           mean_{mean_merit}_
                                           sd_{merit_deviation}_
                                           model_{model_type}_
                                           walls_{wall_stop}_
                                           ppl_{seed_population}""".replace(
            "\n", ""
        ).replace(
            " ", ""
        )
    )

    # Runs the simulation a number of times defined by the 'iterations' argument.
    for y in range(iterations):
        # Names and numbers the output file.
        file_name = file_name_basic + f"__model_run={y+1}"

        # Creates agents.
        agents = sim.setup_merit_agents()

        # Runs the simulation.
        ag = sim.update_agents(
            agents,
            iterations_per_frame,
            agent_speed,
            agent_contact_range,
            contact_range_falloff,
            plane_size,
            1,
            model_type,
            wall_stop,
        )

        # Prepares the array to be saved by sending it to a function that calculates additional prameters.
        agents_final = sim.calculate_and_add_columns(ag)

        # If in a saved folder there are simulations with the same name (which means: the same model parameters) and same numbering
        # then it finds the first free number for the saved file.
        while f"{file_name}.npy" in datafile_list:
            saved_file_number = int("".join(list(filter(str.isdigit, file_name[-4:]))))
            new_number = saved_file_number + 1
            file_name = file_name[:-4] + file_name[-4:].replace(
                str(saved_file_number), str(new_number)
            )

        # Sets the final file name and path.
        saved_file_name = file_name
        final_path = mypath + "/" + saved_file_name

        # Saves the array as a npy file.
        np.save(final_path, agents_final)
        print("Saved file as: ", saved_file_name)


def load_multiple(partial_load=False, partial_number=100, search_string="", mypath=""):
    """
    Loads multiple saved arrays as a 3-dimensional numpy array. When activated it catalogues files in a folder defined by
    'mypath' argument or - if that is left blank - in a "SAVED_SIMS" subfolder of the current folder.
    It then prints out a list of all saved models found, grouping them by their parameters and providing information on
    how many runs of the model are saved for each configuration of the model. After choosing which configuration to load
    it loads and returns them.

    :param partial_load    (bool) : should the loader load a limited number of files with chosen model parameters.
    :param partial_number   (int) : how many files should be loaded if partial_load is True.
    :param search_string (string) : when showing what files can be loaded, the loader will display only
                                    files containing search term defined here.
    :param mypath        (string) : the path where loader will look for files. If left in default mode it will search
                                    the "SAVED_SIMS" subfolder of the current folder.
    :returns agents_loaded  (numpy array) : 3d numpy array containing all the model runs of a chosen type.
    : returns agents_description (string) : text describing the configuration of the model.
    """
    # If mypath argument is empty, creates one based on a current folder.
    if mypath == "":
        mypath = os.getcwd() + "/SAVED_SIMS"

    # Indexes files in the save folder.
    datafile_list = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    # Creates lists for the loaded files and data processing.
    load_list = []
    agents_loaded = []
    set_list = []
    index_list = []

    # Strips saved file names of their numbering and file extension.
    strip_list = [x.partition("__")[0] for x in datafile_list]

    # Creates a temporary list of model configurations found in the file list.
    set_list_temp = [x for x in set(strip_list)]

    # Creates a final list of configurations by discarding those that do not contain a search term.
    for x in range(len(set_list_temp)):
        if search_string in set_list_temp[x]:
            set_list.append(set_list_temp[x])

    # Sorts the configuration list.
    set_list.sort()

    # Creates an index list, providing indexes of first occurences of types of model configurations in file lists.
    for x in range(len(set_list)):
        for y in range(len(strip_list)):
            if strip_list[y] == set_list[x]:
                index_list.append(y)
                break

    # Counts how many saved files are there for each of the configurations.
    file_count = [0 for x in set_list]

    for x in range(len(set_list)):
        for y in range(len(strip_list)):
            if strip_list[y] == set_list[x]:
                file_count[x] += 1

    # Prints the list of configurations.
    print("Data list:\n")
    for x in range(len(set_list)):
        print(f"plik {x} ({file_count[x]} files): {set_list[x]}\n")

    # Asks what configuration is to be loaded.
    Chosen_file = input("Which file should be loaded?")

    if int(Chosen_file) < (len(set_list)):
        load_file_name = datafile_list[index_list[int(Chosen_file)]]
    else:
        print("Wrong number\n")
        return "Error!", "Error!"

    # Creates a list of files to be loaded.
    for x in range(len(datafile_list)):
        if datafile_list[x].partition("=")[0] == load_file_name.partition("=")[0]:
            load_list.append(datafile_list[x])

    if partial_load == True:
        load_list = load_list[:partial_number]

    # Loads the files and puts them into a python list.
    for y in range(len(load_list)):
        final_path = mypath + "/" + load_list[y]
        print(
            f"Loading: {round(((y+1)/(len(load_list)+1)) * 100)} %",
            end="\r",
            flush=True,
        )
        loaded_array = np.load(final_path)
        agents_loaded.append(loaded_array)

    print(f"\n{len(load_list)} files loaded!")

    # Transforms a list of 2d arrays into a 3d array, and creates string describing the model.
    agents_loaded = np.array(agents_loaded)
    agents_description = datafile_list[index_list[int(Chosen_file)]].partition("__")[0]

    return (agents_loaded, agents_description)


def show_new_agents(agents):
    """
    Changes agents numpy array into pandas DataFrame. In the process it discards some columns (id and movement information).

    :param agents      (nupy array)       : an array containing agents information.
    :returns agents_df (pandas DataFrame) : DataFrame with the same information.
    """
    # Establishes if the agent data was modified by the calculate_and_add_columns class function, and names columns accordingly.
    if agents.shape[1] > 15:
        agents_df = pd.DataFrame(
            agents,
            columns=[
                "id",
                "position_x",
                "position_y",
                "x_move",
                "y_move",
                "agent_size",
                "agent_merit",
                "agent_accumulation",
                "state",
                "production_time",
                "starting_x",
                "starting_y",
                "dist_to_nearest",
                "nearest_ag_merit",
                "nearest_to_nearest",
                "distance to a wall",
            ],
        )
    else:
        agents_df = pd.DataFrame(
            agents,
            columns=[
                "id",
                "position_x",
                "position_y",
                "x_move",
                "y_move",
                "agent_size",
                "agent_merit",
                "agent_accumulation",
                "state",
                "production_time",
                "starting_x",
                "starting_y",
            ],
        )

    # Generates the DataFrame.
    agents_df = agents_df.drop(columns=["id", "x_move", "y_move"])

    return agents_df
