import math
import numpy as np
import random as rnd

# criar um novo elemento a partir de uma copia (com referencia e etc independente)
from copy import deepcopy

# para calcular a fitness
from sklearn.metrics import mean_squared_error as mse


class GeneticProgramming:
    
    def __init__(self, data, functions, target, population_size, individual_size, fitness_method, 
                 fitness_penalty, tournament_size, crossover_prob, mutation_prob, mutation_nodes, elitism_size,
                 max_generations, populate_method):
        
        self.data                = data
        self.terminals           = list(data.columns)
        
        self.functions           = functions
        self.target              = target
        self.population_size     = population_size
        self.individual_size     = individual_size
        self.fitness_method      = fitness_method
        self.fitness_penalty     = fitness_penalty
        self.tournament_size     = tournament_size
        self.crossover_prob      = crossover_prob
        self.mutation_prob       = mutation_prob
        self.mutation_nodes      = mutation_nodes
        self.elitism_size        = elitism_size
        self.max_generations     = max_generations
        self.populate_method     = populate_method
        
    #### INITIAL POPULATION
    def generate_tree(self, max_size, method, size = 1):
        terminals = self.terminals
        functions = self.functions
        
        if size == max_size:
            chosen_feature = rnd.randint(0, len(terminals) - 1)
            node           = {"terminal": terminals[chosen_feature]}
            
        else:
            if method == "grow":
                #escolher aleatoriamente entre terminal ou função
                chosen_node = rnd.choice(terminals + list(functions)) 

            elif method == "full":
                chosen_node = rnd.choice(list(functions)) 
            
            if "function" in chosen_node: # se for uma função      
                indv_function  = chosen_node["function"]
                indv_phenotype = chosen_node["phenotype"]
                indv_children  = [self.generate_tree(max_size, method, size + 1) 
                                  for _ in range(chosen_node["children"])]   

                node = {"function": indv_function, "children": indv_children, "phenotype": indv_phenotype}

            else:                                 
                chosen_feature = rnd.randint(0, len(terminals) - 1)
                node           = {"terminal": terminals[chosen_feature]}

        return node


    def generate_initial_population(self):
        population_size = self.population_size
        max_size        = self.individual_size
        populate_method = self.populate_method
        
        population = []
        
        if populate_method == "grow" or populate_method == "full":

            while len(population) != population_size:
                individual = self.generate_tree(max_size, method)
                
                population.append(individual)

        elif populate_method == "ramped":
            
            groups = math.ceil(population_size / (max_size - 1)) # arredonda pra cima            
            
            for D in range(2, groups + 1): # grupos (2, ..., D)
                
                half = math.ceil(groups / 2)
                
                grow_pop = [self.generate_tree(D, method = "grow") for _ in range(half)]
                full_pop = [self.generate_tree(D, method = "full") for _ in range(half)]
                
                # como arredonda pra cima, se gerar mais do que o que se espera do grupo,
                # remove aleatoriamente ate atingir o tamanho
                this_group = grow_pop + full_pop
                group_len  = (len(grow_pop) + len(full_pop))
                
                to_remove = group_len - groups
                
                if to_remove > 0:                     
                    for x in range(to_remove):
                        selection = rnd.randint(0, len(this_group) - 1)
                        this_group.pop(selection)
                
                population = population + this_group
            
            to_remove = len(population) - population_size
            
            if to_remove > 0:                
                for x in range(to_remove):
                    selection = rnd.randint(0, len(population) - 1)
                    population.pop(selection)
                      
        else:
            raise ValueError("To generate initial population you can use 'grow', 'full' or 'ramped'.")

        return population
    
    
    
    #### FITNESS
    ### read the expression
    # individual['func'] diz qual a função que deve ler, vai empilhando até achar terminal 
    # e aplica a função, desempillhando
    def calculate_expression(self, individual, row):
        if "children" not in individual:      # se o nó não tiver filho (i.e., for um terminal)                                                                 
            return row[individual["terminal"]]

        else:
            return individual["function"](*[self.calculate_expression(children, row) 
                                            for children in individual["children"]]) 

        
    def get_size(self, individual):
        if "children" not in individual:                                                                          
            return 1     

        else:
            return 1 + max([self.get_size(children) for children in individual["children"]])
        
    
    def get_prediction(self, individual):
        prediction = [self.calculate_expression(individual, row) for _, row in self.data.iterrows()]
        return prediction


    def get_fitness(self, individual):  
        target         = self.target
        fitness_method = self.fitness_method
        penalty        = self.fitness_penalty
        
        prediction     = self.get_prediction(individual)
        
        if fitness_method == "mse":
            #val = ((prediction - target.values) ** 2).mean()
            val = mse(target.values, prediction, squared = True)
            
            #squared: boolean value, optional (default = True)
            # If True returns MSE value, if False returns RMSE value.

        elif fitness_method == "rmse":
            #val = np.sqrt(((prediction - target.values) ** 2).mean())
            val = mse(target.values, prediction, squared = False)

        elif fitness_method == "sae":
            val = np.sum(np.abs(prediction - target.values))

        else:
            raise ValueError("Fitness not implemented. Try again!")

        # a ideia é penalizar modelos complexos demais aumentando o valor da fitness
        # penalty deve estar entre 0 e 1
        # se penalty = 0, penalty_fit = 1 e val = val * 1 (ou seja, nao altera)
        # se penalty = 1, penalty_fit = get_size e val = val * get_size (muito alto)
        if penalty == 0:
            val = val
        
        elif 0 < penalty <= 1:
            n_individual = self.get_size(individual)
            
            if n_individual > self.individual_size:
                penalty_fit = n_individual ** penalty
                val         = val + penalty_fit

        else: 
            raise ValueError("Penalty value must be in the interval [0, 1]")

        return val
    
    
    #### TOURNAMENT
    # um subconjunto de k individuos é retirado aleatoriamente da população
    # e o melhor individuo desse subconjunto é selecionado
    # quanto maior k, maior a pressao seletiva
    def tournament(self, population_members, population_fitness):
        tournament_size = self.tournament_size
        
        warriors                   = [rnd.randint(0, len(population_members) - 1) for _ in range(tournament_size)]
        warriors_fitness           = [(population_fitness[i], population_members[i]) for i in warriors]
        chosen_fitness, chosen_one = min(warriors_fitness, key = lambda x: x[0])

        return chosen_fitness, chosen_one
    
    
    
    
    def get_child_fitness(self, p1_fitness, p2_fitness, child):
        # media da fitness dos pais
        mean_parents_fitness = (p1_fitness + p2_fitness) / 2
        
        child_fitness        = self.get_fitness(child)

        if child_fitness > mean_parents_fitness:
            child_value = 'better'

        elif child_fitness == mean_parents_fitness:
            child_value = 'equal'

        else:
            child_value = 'worse'

        return child_value
            
    
    #### OPERATORS: CROSSOVER AND MUTATION
    def selection(self, population_members, population_fitness):
        crossover_prob = self.crossover_prob
        mutation_prob  = self.mutation_prob
    
        p1_fitness, parent1 = self.tournament(population_members, population_fitness)
        
        child         = parent1
        child_value   = 'none'
        
        # gera numero entre 0 e 1
        if rnd.random() < crossover_prob:
            p2_fitness, parent2 = self.tournament(population_members, population_fitness)
            child               = self.crossover(parent1, parent2)
                        
            child_value = self.get_child_fitness(p1_fitness, p2_fitness, child)
                        
        if rnd.random() < mutation_prob:
            child = self.mutation(child)
        
        return child_value, child


    ### crossover
    def crossover(self, parent1, parent2):
        child = deepcopy(parent1)
        
        if "children" not in parent1 and "children" in parent2:
            cut_parent2 = self.get_cut_point(parent2, None, 0)
            child       = cut_parent2
            
        elif "children" in parent1 and "children" not in parent2:
            cut_parent1 = self.get_cut_point(child, None, 0)
            child       = cut_parent1
            
        elif "children" in parent1 and "children" in parent2: # todos dois mais que uma folha
            cut_parent1 = self.get_cut_point(child, None, 0)
            cut_parent2 = self.get_cut_point(parent2, None, 0)

            chosen_child = len(cut_parent1["children"]) # qual filho do no
            to_change    = rnd.randint(0, chosen_child - 1)

            cut_parent1["children"][to_change] = cut_parent2
            
        else:
            child = rnd.choice([parent1, parent2])

        return child
    
    ### mutation
    # mutation corta em um ponto e muda esse ponto por uma expressao aleatoria
    def mutation(self, parent1):
        mutant   = deepcopy(parent1)
        max_size = rnd.randint(1, self.mutation_nodes)
        method   = rnd.choice(["full", "grow"])
        
        mutant_nodes = self.generate_tree(max_size, method)
        
        if "children" in mutant:
            cut_point    = self.get_cut_point(mutant, None, 0)
            to_cut       = len(cut_point["children"])
            chosen_nodes = rnd.randint(0, to_cut - 1)
            
            cut_point["children"][chosen_nodes] = mutant_nodes
            
        else:
            cut_point = mutant_nodes

        return mutant


    ### copiar toda a arvore a partir de um ponto
    def get_cut_point(self, node, parent, size):
        if "children" not in node: # se é folha
            return parent

        else:
            chosen_node = len(node["children"]) # qual filho do nó
            cut_point   = rnd.randint(0, chosen_node - 1)
            
            return self.get_cut_point(node["children"][cut_point], node, size + 1)
        
        
    def elitism(self, population_members, population_fitness):
        k = self.elitism_size
        
        # sort population based on fitness
        sorted_population = [x for _, x in sorted(zip(population_fitness, population_members), 
                                                  key = lambda pair: pair[0])]
        
        return sorted_population[:k]
    
    
    def get_new_population(self, population_members, population_fitness):    
        elitism_size = self.elitism_size
        
        population     = []
        children_value = []
    
        population_size  = len(population_members) - elitism_size            
            
        while len(population) != population_size:
            child_value, new_individual = self.selection(population_members, population_fitness)
            
            # numero de individuos gerados por crossover melhores e piores que a fitness média dos pais
            children_value.append(child_value)
            population.append(new_individual)
     
                      
        if elitism_size != 0:
            individual_elite = self.elitism(population_members, population_fitness)
            population       = population + individual_elite     # append duas listas
        
        return children_value, population



########### FUNÇÕES AUXILIARES
    def print_individual(self, individual):
        if "children" not in individual:    # se o nó não tiver filho (i.e., for um terminal)  
            return individual["terminal"]

        else:
            return individual["phenotype"].format(*[self.print_individual(children) for children in individual["children"]])

    

    def get_repeated_individuals(self, population):
        phenotype_population = [self.print_individual(ind) for ind in population]
        
        n_population      = len(phenotype_population)
        n_pop_without_rep = len(set(phenotype_population))
        
        return n_population - n_pop_without_rep
    
    
    def get_population_size(self, population):
        # conta o tamanho dos individuos de uma população
        pop_size = [self.get_size(ind) for ind in population]
        
        return pop_size
    
    
#     def check_terminate_condition(self, generation = None, fitness = None):
#         condition = self.terminate_condition
        
#         if condition == "max_generations":
#             if generation == self.max_generations + 1:
#                 return False
#             else:
#                 return True
            
#         elif condition == 'termination':
#             if best_fitness <= self.terminate_value:
#                 return False
#             else:
#                 return True
        
#         else:
#             raise ValueError("The termination conditions so far are 'max_generations' and 'termination'.")
        
        
    def run_GP(self):
        global_fitness     = {}
        best_fitness       = float("inf") 
        repeated_solutions = {}
        crossover_fitness  = {}
        population_size    = {}
        
        population = self.generate_initial_population()
        
        for n_gen in range(1, self.max_generations + 1):
            print("-------- GENERATION N {}".format(n_gen))
            
            fitness_values = []
            
            repeated_solutions[n_gen] = self.get_repeated_individuals(population)
            population_size[n_gen]    = self.get_population_size(population)
            
            for individual in population:  
                score = self.get_fitness(individual)
                fitness_values.append(score)

                if score < best_fitness:                                                                      
                    best_fitness    = score                                                                 
                    best_individual = individual
                    
                global_fitness[n_gen] = fitness_values

            children_value, population = self.get_new_population(population, fitness_values)
            
            crossover_fitness[n_gen] = children_value

        aux1 = "Best fitness: {:.02f}".format(best_fitness)
        aux2 = "Best individual: {}".format(self.print_individual(best_individual))

        print(aux1); print(aux2)
        
        print("\n")
        
        return global_fitness, repeated_solutions, best_individual, crossover_fitness, population_size