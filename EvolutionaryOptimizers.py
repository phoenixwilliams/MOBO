import numpy as np
from operator import attrgetter

class Solution:
    def __init__(self, genotype):
        self.genotype = genotype
        self.fitness = None

    def set_fitness(self, f):
        self.fitness = f



def rand_1(population, F):
    idxs = [idx for idx in range(len(population))]
    a, b, c = np.random.choice(idxs, 3, replace=False)
    a, b, c = population[a].genotype, population[b].genotype, population[c].genotype
    mutant = np.clip(a + F * (b - c), 0, 1)
    return Solution(mutant)




def binomial_crossover(parent1, parent2, cr):
    parent1geno, parent2geno = parent1.genotype, parent2.genotype

    child_geno = parent1geno[:]
    dimension = len(child_geno)
    cross_points = np.random.rand(dimension) < cr

    if not np.any(cross_points):
        cross_points[np.random.randint(0, dimension)]

    trial = np.where(cross_points, parent2geno, parent1geno)
    return trial



def evaluate_population(function, population, min_x, max_x):
    for soln in population:
        # map to original space
        orgin = min_x + soln.genotype * (max_x - min_x)
        soln.set_fitness(function(orgin))


class DifferentialEvolution:
    def __init__(self, design):
        self.design = design


    def optimize(self, function):
        min_x, max_x = self.design["bounds"][0], self.design["bounds"][1]
        population = [Solution(np.random.uniform(0, 1, size=(len(min_x),))) for _ in range(self.design["popsize"])]

        evaluate_population(function, population, min_x, max_x)

        best_soln = min(population, key=attrgetter("fitness"))
        best_solution = min_x + best_soln.genotype * (max_x - min_x)
        best_fitness = best_soln.fitness

        for it in range(self.design["iterations"]):
            for soln in population:
                mutant = self.design["mutant_function"](population, **self.design["mutant_params"])
                trail_geno = binomial_crossover(soln, mutant, self.design["cr"])
                orgin = min_x + trail_geno * abs(max_x - min_x)

                f = function(orgin)

                if f <= soln.fitness:
                    soln.genotype = trail_geno
                    soln.set_fitness(f)

                    if f < best_fitness:
                        best_fitness = f
                        best_solution = orgin


        return best_solution, best_fitness








