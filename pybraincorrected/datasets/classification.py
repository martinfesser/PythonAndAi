from __future__ import print_function

__author__ = "Martin Felder, felder@in.tum.de"
#modified version to allow splitwithproportion returning a ClassificationDataSet 

from numpy import zeros, where, ravel, random
from pybrain.datasets import SupervisedDataSet

class ClassificationDataSet(SupervisedDataSet):
    """ Specialized data set for classification data. Classes are to be numbered from 0 to nb_classes-1. """

    def __init__(self, inp, target=1, nb_classes=0, class_labels=None):
        """Initialize an empty dataset.

        `inp` is used to specify the dimensionality of the input. While the
        number of targets is given by implicitly by the training samples, it can
        also be set explicity by `nb_classes`. To give the classes names, supply
        an iterable of strings as `class_labels`."""
        # FIXME: hard to keep nClasses synchronized if appendLinked() etc. is used.
        SupervisedDataSet.__init__(self, inp, target)
        self.addField('class', 1)
        self.nClasses = nb_classes
        if len(self) > 0:
            # calculate class histogram, if we already have data
            self.calculateStatistics()
        self.convertField('target', int)
        if class_labels is None:
            self.class_labels = list(set(self.getField('target').flatten()))
        else:
            self.class_labels = class_labels
        # copy classes (may be changed into other representation)
        self.setField('class', self.getField('target'))


    @classmethod
    def load_matlab(cls, fname):
        """Create a dataset by reading a Matlab file containing one variable
        called 'data' which is an array of nSamples * nFeatures + 1 and
        contains the class in the first column."""
        from mlabwrap import mlab #@UnresolvedImport
        d = mlab.load(fname)
        return cls(d.data[:, 0], d.data[:, 1:])

    @classmethod
    def load_libsvm(cls, f):
        """Create a dataset by reading a sparse LIBSVM/SVMlight format file
        (with labels only)."""
        nFeat = 0
        # find max. number of features
        for line in f:
            n = int(line.split()[-1].split(':')[0])
            if n > nFeat:
                nFeat = n
        f.seek(0)
        labels = []
        features = []
        # read all data
        for line in f:
            # format is:
            # <class>  <featnr>:<featval>  <featnr>:<featval> ...
            # (whereby featnr starts at 1)
            if not line: break
            line = line.split()
            label = int(line[0])
            feat = []
            nextidx = 1
            for r in line[1:]:
                # construct list of features, taking care of sparsity
                (idx, val) = r.split(':')
                idx = int(idx)
                for _ in range(nextidx, idx):
                    feat.append(0.0)
                feat.append(float(val))
                nextidx = idx + 1
            for _ in range(nextidx, nFeat + 1):
                feat.append(0.0)
            features.append(feat[:])    # [:] causes copy
            labels.append([label])

        DS = cls(features, labels)
        return DS

    def __add__(self, other):
        """Adds the patterns of two datasets, if dimensions and type match."""
        if type(self) != type(other):
            raise TypeError('DataSets to be added must agree in type')
        elif self.indim != other.indim:
            raise TypeError('DataSets to be added must agree in input dimensions')
        elif self.outdim != 1 or other.outdim != 1:
            raise TypeError('Cannot add DataSets in 1-of-k representation')
        elif self.nClasses != other.nClasses:
            raise IndexError('Number of classes does not agree')
        else:
            result = self.copy()
            for pat in other:
                result.addSample(*pat)
            result.assignClasses()
        return result

    def assignClasses(self):
        """Ensure that the class field is properly defined and nClasses is set.
        """
        if len(self['class']) < len(self['target']):
            if self.outdim > 1:
                raise IndexError('Classes and 1-of-k representation out of sync!')
            else:
                self.setField('class', self.getField('target').astype(int))

        if self.nClasses <= 0:
            flat_labels = list(ravel(self['class']))
            classes = list(set(flat_labels))
            self.nClasses = len(classes)

    def calculateStatistics(self):
        """Return a class histogram."""
        self.assignClasses()
        self.classHist = {}
        flat_labels = list(ravel(self['class']))
        for class_ in range(self.nClasses):
            self.classHist[class_] = flat_labels.count(class_)
        return self.classHist

    def getClass(self, idx):
        """Return the label of given class."""
        try:
            return self.class_labels[idx]
        except IndexError:
            print("error: classes not defined yet!")

    def _convertToOneOfMany(self, bounds=(0, 1)):
        """Converts the target classes to a 1-of-k representation, retaining the
        old targets as a field `class`.

        To supply specific bounds, set the `bounds` parameter, which consists of
        target values for non-membership and membership."""
        if self.outdim != 1:
            # we already have the correct representation (hopefully...)
            return
        if self.nClasses <= 0:
            self.calculateStatistics()
        oldtarg = self.getField('target')
        newtarg = zeros([len(self), self.nClasses], dtype='Int32') + bounds[0]
        for i in range(len(self)):
            newtarg[i, int(oldtarg[i])] = bounds[1]
        self.setField('target', newtarg)
        self.setField('class', oldtarg)
        # probably better not to link field, otherwise there may be confusion
        # if getLinked() is called?
        ##self.linkFields(self.link.append('class'))

    def _convertToClassNb(self):
        """The reverse of _convertToOneOfMany. Target field is overwritten."""
        newtarg = self.getField('class')
        self.setField('target', newtarg)

    def __reduce__(self):
        _, _, state, _lst, _dct = super(ClassificationDataSet, self).__reduce__()
        creator = self.__class__
        args = self.indim, self.outdim, self.nClasses, self.class_labels
        return creator, args, state, iter([]), iter({})

    def splitByClass(self, cls_select):
        """Produce two new datasets, the first one comprising only the class
        selected (0..nClasses-1), the second one containing the remaining
        samples."""
        leftIndices, dummy = where(self['class'] == cls_select)
        rightIndices, dummy = where(self['class'] != cls_select)
        leftDs = self.copy()
        leftDs.clear()
        rightDs = leftDs.copy()
        # check which fields to split
        splitThis = []
        for f in ['input', 'target', 'class', 'importance', 'aux']:
            if self.hasField(f):
                splitThis.append(f)
        # need to synchronize input, target, and class fields
        for field in splitThis:
            leftDs.setField(field, self[field][leftIndices, :])
            leftDs.endmarker[field] = len(leftIndices)
            rightDs.setField(field, self[field][rightIndices, :])
            rightDs.endmarker[field] = len(rightIndices)
        leftDs.assignClasses()
        rightDs.assignClasses()
        return leftDs, rightDs

    def castToRegression(self, values):
        """Converts data set into a SupervisedDataSet for regression. Classes
        are used as indices into the value array given."""
        regDs = SupervisedDataSet(self.indim, 1)
        fields = self.getFieldNames()
        fields.remove('target')
        for f in fields:
            regDs.setField(f, self[f])
        regDs.setField('target', values[self['class'].astype(int)])
        return regDs

    def splitWithProportion(self, proportion = 0.5):
        """Produce two new datasets, the first one containing the fraction given
        by `proportion` of the samples."""
        indicies = random.permutation(len(self))
        separator = int(len(self) * proportion)

        leftIndicies = indicies[:separator]
        rightIndicies = indicies[separator:]

        leftDs = ClassificationDataSet(inp=self['input'][leftIndicies].copy(),
                                   target=self['target'][leftIndicies].copy())
        rightDs = ClassificationDataSet(inp=self['input'][rightIndicies].copy(),
                                    target=self['target'][rightIndicies].copy())
        return leftDs, rightDs



