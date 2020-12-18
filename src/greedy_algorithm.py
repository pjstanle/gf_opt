import numpy as np
from math import log

class GreedyAlgorithm():

    # A couple of discrete optimization algorithms with greedy principles
    def __init__(self):

        # inputs
        self.bits = np.array([]) # array of ints same length as design_variables. 0 signifies an integer design variable
        self.bounds = np.array([]) # array of tuples same length as design_variables
        self.variable_type = np.array([]) # array of strings same length as design_variables ('int' or 'float')
        self.objective_function = None # takes design_variables as an input and outputs the objective values (needs to account for any constraints already)
        
        # internal variables, you could output some of this info if you wanted
        self.design_variables = np.array([])
        self.nbits = 0
        self.nvars = 0
        self.parent_population = np.array([])
        self.offspring_population = np.array([])
        self.parent_fitness = 0.0
        self.offspring_fitness = 0.0
        self.discretized_variables = {} 

        # outputs
        self.solution_history = np.array([])
        self.optimized_function_value = 0.0
        self.optimized_design_variables = np.array([])


    def chromosome_2_variables(self,chromosome):        
        """convert the binary chromosomes to design variable values"""
        first_bit = 0

        float_ind = 0

        for i in range(self.nvars):
            binary_value = 0
            for j in range(self.bits[i]):
                binary_value += chromosome[first_bit+j]*2**j
            first_bit += self.bits[i]

            if self.variable_type[i] == "float":
                self.design_variables[i] = self.discretized_variables["float_var%s"%float_ind][binary_value]
                float_ind += 1

            elif self.variable_type[i] == "int":
                self.design_variables[i] = self.bounds[i][0] + binary_value


    def optimize_greedy(self,initialize="one",print_progress=True):
        # A simple greedy algorithm. Evaluate the objective after switching each bit
        # one at a time. Keep the switched bit that results in the best improvement.
        # Stop after there is no improvement.

        # determine the number of design variables and initialize
        self.nvars = len(self.variable_type)
        self.design_variables = np.zeros(self.nvars)
        float_ind = 0
        for i in range(self.nvars):
            if self.variable_type[i] == "float":
                ndiscretizations = 2**self.bits[i]
                self.discretized_variables["float_var%s"%float_ind] = np.linspace(self.bounds[i][0],self.bounds[i][1],ndiscretizations)
                float_ind += 1

        # determine the total number of bits
        for i in range(self.nvars):
            if self.variable_type[i] == "int":
                int_range = self.bounds[i][1] - self.bounds[i][0]
                int_bits = int(np.ceil(log(int_range,2)+1))
                self.bits[i] = int_bits
            self.nbits += self.bits[i]
        

        # initialize the fitness
        self.parent_fitness = 0.0
        self.offspring_fitness = 0.0

        # initialize the population
        if initialize == "one":
            self.parent_population = np.zeros(self.nbits,dtype=int)
            self.parent_population[0] = 1
            np.random.shuffle(self.parent_population)
        elif initialize=="zero":
            self.parent_population = np.zeros(self.nbits,dtype=int)
        elif initialize=="random":
            done = False
            while done == False:
                self.parent_population = np.random.randint(0,high=2,size=self.nbits)
                self.chromosome_2_variables(self.parent_population)
                self.parent_fitness = self.objective_function(self.design_variables)
                if self.parent_fitness < 1000.0:
                    done = True

        # initialize the offspring population
        self.offspring_population = np.zeros_like(self.parent_population)
        self.offspring_population[:] = self.parent_population[:]

        # initialize the parent fitness
        self.chromosome_2_variables(self.parent_population)
        self.parent_fitness = self.objective_function(self.design_variables)
        
        # initialize optimization
        self.solution_history = np.array([self.parent_fitness])
        converged = False
        best_population = np.zeros(self.nbits)

        if print_progress:
            print("enter greedy loop")
        while converged==False:
            # loop through every bit
            best_fitness = self.parent_fitness
            best_population[:] = self.parent_population[:]
            for i in range(self.nbits):
                self.offspring_population[:] = self.parent_population[:]
                self.offspring_population[i] = (self.parent_population[i]+1)%2

                # check the fitness
                self.chromosome_2_variables(self.offspring_population)
                self.offspring_fitness = self.objective_function(self.design_variables)

                # check the performance, see if it is the best so far
                if self.offspring_fitness < best_fitness:
                    best_fitness = self.offspring_fitness
                    best_population[:] = self.offspring_population[:]
            
            # check convergence
            if best_fitness == self.parent_fitness:
                converged = True
            
            # update values if not converged
            else:
                if print_progress:
                    print(best_fitness)
                self.solution_history = np.append(self.solution_history,best_fitness)
                self.parent_population[:] = best_population[:]
                self.parent_fitness = best_fitness

        # final outputs
        self.optimized_function_value = self.parent_fitness
        self.chromosome_2_variables(self.parent_population)
        self.optimized_design_variables = self.design_variables


    def optimize_switch(self,initialize=0,print_progress=True):

        # determine the number of design variables and initialize
        self.nvars = len(self.variable_type)
        self.design_variables = np.zeros(self.nvars)
        float_ind = 0
        for i in range(self.nvars):
            if self.variable_type[i] == "float":
                ndiscretizations = 2**self.bits[i]
                self.discretized_variables["float_var%s"%float_ind] = np.linspace(self.bounds[i][0],self.bounds[i][1],ndiscretizations)
                float_ind += 1

        # determine the total number of bits
        for i in range(self.nvars):
            if self.variable_type[i] == "int":
                int_range = self.bounds[i][1] - self.bounds[i][0]
                int_bits = int(np.ceil(log(int_range,2)+1))
                self.bits[i] = int_bits
            self.nbits += self.bits[i]
        
        init = True
        while init == True:
        # initialize the population
            if initialize == 0:
                self.parent_population = np.random.randint(0,high=2,size=self.nbits)
            else:
                nones = initialize
                self.parent_population = np.zeros(self.nbits,dtype=int)
                self.parent_population[0:nones] = 1
                np.random.shuffle(self.parent_population)
            self.offspring_population = np.zeros_like(self.parent_population)
            self.offspring_population[:] = self.parent_population[:]

            # initialize the fitness
            self.parent_fitness = 0.0
            self.offspring_fitness = 0.0

            self.chromosome_2_variables(self.parent_population)
            self.parent_fitness = self.objective_function(self.design_variables)
            if self.parent_fitness != 1E6:
                init = False
        
        # initialize the optimization
        converged = False
        converged_counter = 0
        self.solution_history = np.array([self.parent_fitness])
        index = 1
        random_method = 0

        # initialize the order array (determines the order of sweeping through the variables)
        order = np.arange(self.nbits)

        last_solution = self.parent_fitness
       
        while converged==False:
            # check if we've gone through every bit
            ind = index%self.nbits
            if ind == 0:
                # check if there has been any change since the last phase
                if last_solution == self.parent_fitness:
                    converged_counter += 1
                else:
                    last_solution = self.parent_fitness
                    converged_counter = 0

                # check convergence
                if converged_counter >= 3:
                    converged = True

                # shuffle the order array and change the phase
                np.random.shuffle(order)
                random_method = (random_method+1)%3
                if random_method == 0:
                    print("explore")
                if random_method == 1:
                    print("switch row")
                if random_method == 2:
                    print("switch col")

            # set offpring equal to parent

            self.offspring_population[:] = self.parent_population[:]

            # this is the explore phase. Switch a bit, evaluate, and see if we should keep it
            if random_method == 0:
                # switch the value of the appropriate index
                self.offspring_population[order[ind]] = (self.parent_population[order[ind]]+1)%2

                # check the fitness
                self.chromosome_2_variables(self.offspring_population)
                self.offspring_fitness = self.objective_function(self.design_variables)

                # check if we should keep the proposed change
                if self.offspring_fitness < self.parent_fitness:
                    self.solution_history = np.append(self.solution_history,self.offspring_fitness)
                    self.parent_fitness = self.offspring_fitness
                    self.parent_population[:] = self.offspring_population[:]
                    if print_progress:
                        print(self.offspring_fitness)

            # this is the first switch phase, switch adjacent bits (only makes sense if they are arranged spatially in a matrix)
            elif random_method == 1:
                # organize the matrix
                N = int(np.sqrt(len(self.parent_population)))
                M = np.zeros((N,N))
                for i in range(N):
                    M[i,:] = self.offspring_population[i*N:(i+1)*N]

                row = order[ind]%N
                col = int(order[ind]/N)

                # switch adjacent numbers
                t1 = M[row][col]
                t2 = M[(row+1)%N][col]

                if t1 != t2:
                    M[row][col] = t2
                    M[(row+1)%N][col] = t1
                    
                    for i in range(N):
                        self.offspring_population[i*N:(i+1)*N] = M[i][:]
                    # check the fitness
                    self.chromosome_2_variables(self.offspring_population)
                    self.offspring_fitness = self.objective_function(self.design_variables)

                    if self.offspring_fitness < self.parent_fitness:
                        self.solution_history = np.append(self.solution_history,self.offspring_fitness)
                        self.parent_fitness = self.offspring_fitness
                        self.parent_population[:] = self.offspring_population[:]
                        if print_progress:
                            print(self.offspring_fitness)

            # this is the second switch phase, switch adjacent bits in the other dimension (only makes sense if they are arranged spatially in a matrix)
            elif random_method == 2:
                # organize the matrix
                N = int(np.sqrt(len(self.parent_population)))
                M = np.zeros((N,N))
                for i in range(N):
                    M[i][:] = self.offspring_population[i*N:(i+1)*N]

                row = order[ind]%N
                col = int(order[ind]/N)

                # switch adjacent numbers
                t1 = M[row][col]
                t2 = M[row][(col+1)%N]

                if t1 != t2:
                    M[row][col] = t2
                    M[row][(col+1)%N] = t1
                    
                    for i in range(N):
                        self.offspring_population[i*N:(i+1)*N] = M[i][:]
                    # check the fitness
                    self.chromosome_2_variables(self.offspring_population)
                    self.offspring_fitness = self.objective_function(self.design_variables)

                    if self.offspring_fitness < self.parent_fitness:
                        self.solution_history = np.append(self.solution_history,self.offspring_fitness)
                        self.parent_fitness = self.offspring_fitness
                        self.parent_population[:] = self.offspring_population[:]
                        if print_progress:
                            print(self.offspring_fitness)

            # increment the counter
            index += 1

        # final output values
        self.optimized_function_value = self.solution_history[-1]
        self.chromosome_2_variables(self.parent_population)
        self.optimized_design_variables = self.design_variables



if __name__=="__main__":

    def simple_obj(x):
        return x[0]+x[1]

    def rosenbrock_obj(x):
        return (1-x[0])**2 + 100.0*(x[1]-x[0]**2)**2

    def ackley_obj(x):
        p1 = -20.0*np.exp(-0.2*np.sqrt(0.5*(x[0]**2+x[1]**2)))
        p2 = np.exp(0.5*(np.cos(2.*np.pi*x[0]) + np.cos(2.0*np.pi*x[1]))) + np.e + 20.0
        return p1-p2

    def rastrigin_obj(x):
        A = 10.0
        n = len(x)
        tot = 0
        for i in range(n):
            tot += x[i]**2 - A*np.cos(2.0*np.pi*x[i])
        return A*n + tot


    import matplotlib.pyplot as plt

    # from mpl_toolkits.mplot3d import Axes3D
    # X = np.arange(-5, 5, 0.02)
    # Y = np.arange(-5, 5, 0.02)
    # X, Y = np.meshgrid(X, Y)
    # Z = np.zeros_like(X)
    # for i in range(np.shape(Z)[0]):
    #     for j in range(np.shape(Z)[1]):
    #         Z[i][j] = rastrigin_obj(np.array([X[i][j],Y[i][j]]))
    
    # # Plot the surface.
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X, Y, Z,linewidth=0, antialiased=False)

    # plt.show()

    # ga = GeneticAlgorithm()
    # # ga = GreedyAlgorithm()
    # ga.bits = np.array([8,8])
    # ga.bounds = np.array([(-5.0,5.),(-5.,5.0)])
    # ga.variable_type = np.array(["float","float"])
    # ga.population_size = 40
    # ga.max_generation = 1000
    # ga.objective_function = rastrigin_obj
    # ga.crossover_rate = 0.1
    # ga.mutation_rate = 0.01
    # ga.convergence_iters = 25
    # ga.tol = 1E-8

    # ga.optimize_ga(print_progress=False)
    # # ga.optimize_switch()
    # print("optimal function value: ", ga.optimized_function_value)
    # print("optimal design variables: ", ga.optimized_design_variables)
    # print("nbits: ", ga.nbits)
    # plt.plot(ga.solution_history)
    # plt.show()
    
