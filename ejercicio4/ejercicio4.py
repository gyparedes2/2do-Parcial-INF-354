import numpy as np
import random
from deap import base, creator, tools, algorithms
import pandas as pd
import array  # Se añade la importación de 'array'

def LeerMapaDistancia(file_path):
    df = pd.read_csv(file_path, skiprows=0, low_memory=False)
    return np.array(df.values)

def Evaluar(individual, distancia_map):
    distancia = distancia_map[individual[-1]][individual[0]]
    for gene1, gene2 in zip(individual[:-1], individual[1:]):
        distancia += distancia_map[gene1][gene2]
    return distancia,

def CrearHerramienta(ind_size, distancia_map):  
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

    CajaHerramienta = base.Toolbox()
    CajaHerramienta.register("indices", random.sample, range(ind_size), ind_size)
    CajaHerramienta.register("individual", tools.initIterate, creator.Individual, CajaHerramienta.indices)
    CajaHerramienta.register("population", tools.initRepeat, list, CajaHerramienta.individual)
    CajaHerramienta.register("mate", tools.cxPartialyMatched)
    CajaHerramienta.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    CajaHerramienta.register("select", tools.selTournament, tournsize=3)
    CajaHerramienta.register("evaluate", Evaluar, distancia_map=distancia_map)

    return CajaHerramienta

random.seed(169)
pop_size = 100
ind_size = 5
generations = 100
file_path = 'grafo.csv'

distancia_map = LeerMapaDistancia(file_path)
    
CajaHerramienta = CrearHerramienta(ind_size, distancia_map)
pop = CajaHerramienta.population(n=pop_size)

hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("promedio", np.mean)
stats.register("DesvEstandar", np.std)
stats.register("ValMinimo", np.min)
stats.register("ValMaximo", np.max)
    
algorithms.eaSimple(pop, CajaHerramienta, 0.7, 0.2, generations, stats=stats, halloffame=hof)
    
print(hof)
print('B, D, A, C, E')
print("Mejor distancia individual:", Evaluar(hof[0], distancia_map)[0])
