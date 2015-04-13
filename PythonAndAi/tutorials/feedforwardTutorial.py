from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules.tanhlayer import TanhLayer
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.supervised.trainers.mixturedensity import BackpropTrainerMix
from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer
from pybrain.structure.connections.full import FullConnection

def printNetResult(identifier, net):
    print(identifier, net.activate((0, 0)), net.activate((0, 1)), net.activate((1, 0)), net.activate((1, 1)))    

ds = SupervisedDataSet(2,1)

ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))

for input, target in ds:
    print(input, target)
    
#define layers and connections
inLayer = LinearLayer(2)
hiddenLayerOne = SigmoidLayer(4, "one")
hiddenLayerTwo = SigmoidLayer(4, "two")
outLayer = LinearLayer(1)
inToHiddenOne = FullConnection(inLayer, hiddenLayerOne)
hiddenOneToTwo = FullConnection(hiddenLayerOne, hiddenLayerTwo)
hiddenTwoToOut = FullConnection(hiddenLayerTwo, outLayer)

#wire the layers and connections to a net
net = FeedForwardNetwork()
net.addInputModule(inLayer)
net.addModule(hiddenLayerOne)
net.addModule(hiddenLayerTwo)
net.addOutputModule(outLayer)
net.addConnection(inToHiddenOne)
net.addConnection(hiddenOneToTwo)
net.addConnection(hiddenTwoToOut)
net.sortModules()

print(net)

trainer = BackpropTrainer(net, ds)

for i in range(20):
    for j in range(1000):               
        trainer.train()
    printNetResult(i, net)
    
print(net)
print(inToHiddenOne.params)
print(hiddenOneToTwo.params)
print(hiddenTwoToOut.params)
