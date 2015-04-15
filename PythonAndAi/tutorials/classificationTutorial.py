from scipy import diag, arange, meshgrid, where

from numpy.random.mtrand import multivariate_normal
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules.softmax import SoftmaxLayer
from pybrain.supervised.trainers.backprop import BackpropTrainer
from builtins import range
from pybrain.utilities import percentError
from pylab import figure, plot, ioff, clf, hold, contourf, ion, draw, show
from pybraincorrected.datasets.classification import ClassificationDataSet


means = [(-1, 0), (2, 4), (3, 1)]
cov = [diag([1, 1]), diag([0.5, 1.2]), diag([1.5, 0.7])]
alldata = ClassificationDataSet(2, 1, nb_classes=3)

for n in range(400):
    for klass in range(3):
        input_ = multivariate_normal(means[klass], cov[klass])
        alldata.addSample(input_, [klass])
        #print(n, klass, input_)

testdata, traindata = alldata.splitWithProportion(0.25)
testdata._convertToOneOfMany()
traindata._convertToOneOfMany()

print("data dimensions ", len(traindata), traindata.indim, testdata.indim)
print("sample test data input", traindata['input'][0])
print("sample test data target", traindata['target'][0])
print("sample test data class",  traindata["class"][0])

#for key in traindata.data:
#    print(key)

fnn = buildNetwork(traindata.indim, 5, traindata.outdim, outclass = SoftmaxLayer)

trainer = BackpropTrainer(fnn, dataset=traindata, momentum=0.1, verbose=True, weightdecay=0.01)

ticks = arange(-3., 6., 0.2)
X, Y = meshgrid(ticks, ticks)

griddata = ClassificationDataSet(2, 1, nb_classes=3)
for i in range(X.size):
    griddata.addSample([X.ravel()[i],Y.ravel()[i]], [0])

griddata._convertToOneOfMany()

for i in range(20):
    trainer.trainEpochs(1) # usually 5
    
trainresult = percentError(trainer.testOnClassData(), traindata["class"])
testresult = percentError(trainer.testOnClassData(), testdata["class"])

print("epoch %4d" % trainer.totalepochs, "trainerror %5.2f%%" % trainresult, "testerror %5.2f%%" % testresult)
    
out = fnn.activateOnDataset(griddata)
out = out.argmax(axis=1)
out = out.reshape(X.shape)

figure(1)
 # might be the wrong import for the following lines
ioff()  
clf()
hold(True)
for c in [0,1,2]:
    here, _ = where(testdata["class"]==c)
    plot(testdata["input"][here, 0], testdata["input"][here, 1], 'o')
if out.max()!=out.min():
    contourf(X, Y, out)
ion()
draw()

ioff()
show()

