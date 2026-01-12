import operator
import random
import numpy as np
import pandas as pd
from deap import base, creator, gp, tools, algorithms

data = pd.read_csv(
    "ResidentWorkingPersonsAged15YearsandOverbyUsualModeofTransporttoWorkAgeGroupandSexGeneralHouseholdSurvey2015.csv"
)

data = data.select_dtypes(include=[np.number]).dropna()
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

pset = gp.PrimitiveSet("MAIN", X.shape[1])
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)

def protected_div(a, b):
    return a / b if b != 0 else 1

pset.addPrimitive(protected_div, 2)
pset.addEphemeralConstant("rand", lambda: random.uniform(-1, 1))

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def eval_gp(individual):
    func = toolbox.compile(expr=individual)
    preds = []
    for row in X:
        try:
            preds.append(func(*row))
        except:
            preds.append(0)
    mse = np.mean((np.array(preds) - y) ** 2)
    return mse,

toolbox.register("evaluate", eval_gp)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=len, max_value=15))
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=15))

def run_gp():
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    pop, log = algorithms.eaSimple(
        pop, toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=30,
        stats=stats,
        halloffame=hof,
        verbose=True
    )

    return hof[0], log
