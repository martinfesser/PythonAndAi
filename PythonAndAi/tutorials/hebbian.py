#from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.datasets.unsupervised import UnsupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules.tanhlayer import TanhLayer
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.supervised.trainers.mixturedensity import BackpropTrainerMix
from pybrain.unsupervised.trainers.deepbelief import DeepBeliefTrainer


def printNetResult(identifier, net):
    print(identifier, net.activate((0, 0)), net.activate((0, 1)), net.activate((1, 0)), net.activate((1, 1)))    

ds = UnsupervisedDataSet(3)


tf = open('novelty_plant.txt','r')

first = True

for line in tf.readlines():
	if (not first):
		data = [float(x) for x in line.strip().split('\t') if x != '']
		#    indata =  tuple(data[:6])
		#    outdata = tuple(data[6:])
		ds.addSample(data)
	first = False

n = buildNetwork(ds.dim,8,8,1,recurrent=True)
t = DeepBeliefTrainer(n,ds, epochs=50)
t.trainEpochs(1)
t.testOnData(ds, verbose= True)

ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))

for input, target in ds:
    print(input, target)
    
#net = buildNetwork(2, 3, 1, bias=True, hiddenclass=TanhLayer)#1000
# net = buildNetwork(2, 6, 1, bias=True) # 3000
net = buildNetwork(2, 3, 1, bias=True)

trainer = BackpropTrainer(net, ds)

for i in range(20):
    for j in range(1000):               
        trainer.train()
    printNetResult(i, net)