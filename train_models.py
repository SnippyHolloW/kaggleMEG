from scipy.io import loadmat
import numpy as np
from pylab import imshow, show
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 15, 10
import glob, cPickle
from itertools import izip

# get the .mat data from 
# https://www.kaggle.com/c/decoding-the-human-brain/download/train_01_06.zip

def load_and_savez(ifname, ofname):
    ofname = ofname + "_subjects.npz"
    try:
        a = np.load(ofname)
        data = a['X']
        labels = a['y']
        sfreq = a['sfreq']
    except:
        subject_files = glob.glob(ifname)
        print subject_files
        data = None
        labels = None

        for suj in subject_files:
            d = loadmat(suj)
            if data == None:
                data = d['X']
            else:
                data = np.concatenate([data, d['X']], axis=0)
            if labels == None:
                labels = d['y'].ravel()
            else:
                labels = np.concatenate([labels, d['y'].ravel()], axis=0)
        'Used Mem : %.4f Gb' % (data.nbytes / float(2**30))   
        print data.shape
        print labels.shape
        sfreq = d['sfreq']
        np.savez_compressed(ofname, sfreq=sfreq, X=data, y=labels)
    return sfreq, data, labels


def normalize_stupid(data, mean=None, std=None):
    if mean == None:
        mean = data.mean(axis=0)
    if std == None:
        std = data.std(axis=0)
    return (data - mean) / std, mean, std


sfreq, X_train, y_train = load_and_savez("data/train_subject0*.mat", "train")
X_train, mean, std = normalize_stupid(X_train)
print mean.shape
print std.shape
sfreq, X_test, y_test = load_and_savez("data/train_subject1*.mat", "test")
X_test, mean, std = normalize_stupid(X_test, mean, std)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))

#    from sklearn.linear_model import SGDClassifier
#    clf = SGDClassifier(loss="hinge", penalty="l2")
#    #clf.fit(X_train, y_train)
#    #with open("sgd_clf.pkl", "wb") as f:
#    #    cPickle.dump(clf, f)
#    #with open("sgd_clf.pkl", "rb") as f:
#    #    clf = cPickle.load(f)
#
#    #print clf.score(X_test, y_test)
#
#    from lightning.classification import CDClassifier
#    #clf = CDClassifier(penalty="l1/l2",
#    clf = CDClassifier(penalty="l1",
#            loss="squared_hinge",
#            multiclass=False,
#            max_iter=30,
#            alpha=1e-4,
#            C=1.0 / X_train.shape[0],
#            tol=1e-3)
#    clf.fit(X_train, y_train)
#    with open("cd_clf.pkl", "wb") as f:
#        cPickle.dump(clf, f)
#    #with open("cd_clf.pkl", "rb") as f:
#    #    clf = cPickle.load(f)
#    print clf.score(X_test, y_test)


from layers import Linear, ReLU 
from classifiers import LogisticRegression
from nnet_archs import NeuralNet

numpy_rng = np.random.RandomState(123)
nnet = NeuralNet(numpy_rng=numpy_rng, 
        n_ins=X_train.shape[1],
        layers_types=[ReLU, ReLU, LogisticRegression],
        layers_sizes=[100, 100],
        n_outs=1,
        debugprint=0)
train_fn = nnet.get_adadelta_trainer()
epochs = 0
max_epochs = 100
while epochs < max_epochs:
    for x, y in izip(X_train, y_train):
        avg_cost = train_fn(x, y)
        print avg_cost
    

