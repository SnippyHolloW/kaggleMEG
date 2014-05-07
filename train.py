from scipy.io import loadmat
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.kernel_approximation import RBFSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import numpy as np


def train_models(X_train, y_train, X_test, y_test):
    clf = linear_model.SGDClassifier(penalty='elasticnet')
    print clf
    print "fitting a linear elasticnet (L1+L2 regularized linear classif.) with SGD"
    clf = clf.fit(X_train, y_train)
    print "score on the training set", clf.score(X_train, y_train)
    print "score on 80/20 split", clf.score(X_test, y_test)

    rbf_feature = RBFSampler(gamma=1, random_state=1)
    X_train_feats = rbf_feature.fit_transform(X_train)
    X_test_feats = rbf_feature.transform(X_test)
    print "fitting a linear elasticnet with SGD on RBF sampled features"
    clf = clf.fit(X_train_feats, y_train)
    print "score on the training set", clf.score(X_train_feats, y_train)
    print "score on 80/20 split", clf.score(X_test_feats, y_test)

    clf2 = RandomForestClassifier(max_depth=None, min_samples_split=3)
    print clf2
    print "fitting a random forest"
    clf2 = clf2.fit(X_train, y_train)
    print "score on the training set", clf2.score(X_train, y_train)
    print "score on 80/20 split", clf2.score(X_test, y_test)

    clf3 = svm.SVC(kernel='linear')
    print clf3
    print "fitting an SVM with a linear kernel"
    clf3 = clf3.fit(X_train, y_train)
    print "score on the training set", clf3.score(X_train, y_train)
    print "score on 80/20 split", clf3.score(X_test, y_test)

    clf4 = svm.SVC(kernel='rbf')
    print clf4
    print "fitting an SVM with an RBF-kernel"
    clf4 = clf4.fit(X_train, y_train)
    print "score on the training set", clf4.score(X_train, y_train)
    print "score on 80/20 split", clf4.score(X_test, y_test)

    clf5 = linear_model.LogisticRegression(penalty='l1', tol=0.01)
    print clf5
    print "fitting a logistic regression reg. with L1"
    clf5 = clf5.fit(X_train, y_train)
    print "score on the training set", clf5.score(X_train, y_train)
    print "score on 80/20 split", clf5.score(X_test, y_test)


if __name__ == "__main__":
    d = loadmat("data/train_subject01.mat")
    print d.keys()
    print d['X'].shape  # (N_TRIALS, N_ELECTRODES, N_TIMESTAMPS)
    print d['y'].shape  # (N_TRIALS, 1 (label in {0, 1}))
    y = d['y'].ravel()
    print y.shape
    print "================="
    print ">>> random by sampling in the {0, 1} distribution of y is at accuracy:",
    tot = y.sum() * 1. / y.shape[0]
    print max(1-tot, tot)

    tmp = d['X'].shape
    X = d['X'].reshape(tmp[0], tmp[1] * tmp[2])
    print X.shape
    print "================="
    print ">>> as a bag of features without any structure between electrodes nor timesteps"
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)
    train_models(X_train, y_train, X_test, y_test)

    print "================="
    print ">>> same but ONLY AFTER timestep 220"
    X = d['X'][:,:,220:]
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    print X.shape
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)
    train_models(X_train, y_train, X_test, y_test)

    X = d['X']
    tmpMean = np.tile(X.mean(axis=0), (X.shape[0], 1)).reshape(X.shape[0], X.shape[1], X.shape[2])
    X -= tmpMean
    X = X.reshape(tmp[0], tmp[1] * tmp[2])
    print X.shape
    print "================="
    print ">>> as a bag of features without structure but removed mean for each electrode"
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)
    train_models(X_train, y_train, X_test, y_test)

    X = d['X'].sum(axis=2)
    print X.shape
    print "================="
    print ">>> now we're just summing all electrodes along the time axis"
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)
    train_models(X_train, y_train, X_test, y_test)

