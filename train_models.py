from scipy.io import loadmat
import numpy as np
from pylab import imshow, show
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 15, 10
import glob, cPickle
from itertools import izip

# get the .mat data from 
# https://www.kaggle.com/c/decoding-the-human-brain/download/train_01_06.zip

START_FRAME = 125  # take only between 0 (start of the stimulus) and 200ms
END_FRAME = 175
#START_FRAME = 150
#END_FRAME = 175
DROPOUT = True  # Use dropout or not?

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


### TRAIN
sfreq, X_train, y_train = load_and_savez("data/train_subject0*.mat", "train")
# TODO CHANGE
X_train = X_train[:, :, START_FRAME:END_FRAME]
# /TODO CHANGE
# TODO in the load_and_savez
X_train = np.asarray(X_train, dtype='float32')
y_train = np.asarray(y_train, dtype='int32')

X_train, mean, std = normalize_stupid(X_train)
print mean.shape
print std.shape

### VALIDATION
sfreq, X_dev, y_dev = load_and_savez("data/train_subject1[0-3].mat", "dev")
# TODO CHANGE
X_dev = X_dev[:, :, START_FRAME:END_FRAME]
# /TODO CHANGE
X_dev, _, _ = normalize_stupid(X_dev, mean, std)
# TODO in the load_and_savez
X_dev = np.asarray(X_dev, dtype='float32')
y_dev = np.asarray(y_dev, dtype='int32')

### TEST
sfreq, X_test, y_test = load_and_savez("data/train_subject1[4-6].mat", "test")
# TODO CHANGE
X_test = X_test[:, :, START_FRAME:END_FRAME]
# /TODO CHANGE
X_test, _, _ = normalize_stupid(X_test, mean, std)
# TODO in the load_and_savez
X_test = np.asarray(X_test, dtype='float32')
y_test = np.asarray(y_test, dtype='int32')

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
X_dev = X_dev.reshape((X_dev.shape[0], X_dev.shape[1]*X_dev.shape[2]))


### TODO try all of these (uncomment)
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


from layers import ReLU 
from classifiers import LogisticRegression
from nnet_archs import RegularizedNet, DropoutNet, add_fit_and_score
add_fit_and_score(RegularizedNet)
add_fit_and_score(DropoutNet)
from sklearn.metrics import accuracy_score

numpy_rng = np.random.RandomState(123)
# do not forget that the first layer is n_frames * 306 because we're doing it the brute way
# TODO do a few frames by a few frames (not 75*306, more like 5 or 10*306)
# TODO play with L1_reg and L1_reg parameters (also you can set them to 0) independently
# TODO play with architectures (you just need to end by a LogisticRegression)
# TODO play with size of hidden units (50 to 2000)
# TODO add sequentiality / temporality
# TODO adjust features BEFORE here


nnet = None
if DROPOUT:
    nnet = DropoutNet(numpy_rng=numpy_rng, 
            n_ins=X_train.shape[1],
            layers_types=[ReLU, ReLU, LogisticRegression],
            layers_sizes=[1000, 500],
            dropout_rates=[0.4, 0.5, 0.5],  #[0., 0.5, 0.5],
            #layers_types=[ReLU, ReLU, ReLU, LogisticRegression],
            #layers_sizes=[2000, 1000, 1000],
            #dropout_rates=[0.2, 0.5, 0.5, 0.5],
            n_outs=2,
            debugprint=0)
else:
    nnet = RegularizedNet(numpy_rng=numpy_rng, 
            n_ins=X_train.shape[1],
            layers_types=[ReLU, ReLU, LogisticRegression],
            layers_sizes=[2000, 1000],
            n_outs=2,
            L1_reg=1./X_train.shape[0],
            L2_reg=1./X_train.shape[0],
            debugprint=0)

print nnet
nnet.fit(X_train, y_train, X_dev, y_dev, max_epochs=500)  # TODO 1000+ epochs
print nnet.score(X_test, y_test)
with open("nnet_clf.pkl", "wb") as f:
    cPickle.dump(nnet, f)
