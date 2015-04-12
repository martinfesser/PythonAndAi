from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules.tanhlayer import TanhLayer
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.supervised.trainers.mixturedensity import BackpropTrainerMix

def printNetResult(net):
    print(net.activate((0, 0)), net.activate((0, 1)), net.activate((1, 0)), net.activate((1, 1)))    

ds = SupervisedDataSet(2,1)

ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))

for input, target in ds:
    print(input, target)
    
net = buildNetwork(2, 3, 1, bias=True, hiddenclass=TanhLayer)
net = buildNetwork(2, 6, 1, bias=True)

trainer = BackpropTrainer(net, ds)

for i in range(30):
    for j in range(100):               
        trainer.train()
    printNetResult(net)