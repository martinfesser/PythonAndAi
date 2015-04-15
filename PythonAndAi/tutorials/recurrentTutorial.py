from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules.tanhlayer import TanhLayer
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.supervised.trainers.mixturedensity import BackpropTrainerMix
from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer
from pybrain.structure.connections.full import FullConnection
from pybrain.structure.networks.recurrent import RecurrentNetwork

def printNetResult(identifier, net):
    print(identifier, net.activate((0, 0)), net.activate((0, 1)), net.activate((1, 0)), net.activate((1, 1)))    

ds = SupervisedDataSet(2,1)

ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))

for input, target in ds:
    print(input, target)
    

#define net
net = RecurrentNetwork()
net.addInputModule(LinearLayer(2, name="il"))
net.addModule(SigmoidLayer(4, name="h1"))
net.addModule(SigmoidLayer(4, name="h2"))
net.addOutputModule(LinearLayer(1, name="ol"))
c1 = FullConnection(net["il"], net["h1"])
c2 = FullConnection(net["h1"], net["h2"])
c3 = FullConnection(net["h2"], net["ol"])
cr1 = FullConnection(net["h1"], net["h1"])
net.addConnection(c1)
net.addConnection(c2)
net.addConnection(c3)
net.addRecurrentConnection(cr1)                                     
net.sortModules()

print(net)

trainer = BackpropTrainer(net, ds)

for i in range(20):
    for j in range(1000):               
        trainer.train()
    printNetResult(i, net)
    
print(net)
print(net["il"])

print(c1.params)
print(c2.params)
print(c3.params)
print(cr1.params)
