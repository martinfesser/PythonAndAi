from numpy.core.numeric import array
from pybrain.optimization.distributionbased.cmaes import CMAES


def objf(x): return sum(x**2)

x0 = array([2.1, -1])

optimizer = CMAES(objf, x0)
optimizer.minimize = True

optimizer.maxEvaluations = 200

result = optimizer.learn()

print(result) 
