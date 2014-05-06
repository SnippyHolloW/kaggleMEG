from scipy.io import loadmat
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


def train_models(X_train, y_train, X_test, y_test):
    clf = linear_model.SGDClassifier(penalty='elasticnet')
    print clf
    print "fitting a linear elasticnet (L1+L2 regularized linear classif.) with SGD"
    clf.fit(X_train, y_train)
    print "score on 80/20 split", accuracy_score(y_test, clf.predict(X_test))

    clf2 = RandomForestClassifier()
    print clf2
    print "fitting a random forest"
    clf2.fit(X_train, y_train)
    print "score on 80/20 split", accuracy_score(y_test, clf2.predict(X_test)) 

    clf3 = svm.SVC(kernel='linear')
    print clf3
    print "fitting an SVM with a linear kernel"
    clf3.fit(X_train, y_train)
    print "score on 80/20 split", accuracy_score(y_test, clf3.predict(X_test)) 

    clf4 = svm.SVC(kernel='rbf')
    print clf4
    print "fitting an SVM with an RBF-kernel"
    clf4.fit(X_train, y_train)
    print "score on 80/20 split", accuracy_score(y_test, clf4.predict(X_test)) 

    clf5 = linear_model.LogisticRegression(penalty='l1', tol=0.01)
    print clf5
    print "fitting a logistic regression reg. with L1"
    clf5.fit(X_train, y_train)
    print "score on 80/20 split", accuracy_score(y_test, clf5.predict(X_test)) 


if __name__ == "__main__":
    d = loadmat("data/train_subject01.mat")
    print d.keys()
    print d['X'].shape  # (N_TRIALS, N_ELECTRODES, N_TIMESTAMPS)
    print d['y'].shape  # (N_TRIALS, 1 (label in {0, 1}))

    tmp = d['X'].shape
    X = d['X'].reshape(tmp[0], tmp[1] * tmp[2])
    y = d['y'].ravel()
    print X.shape
    print y.shape

    print "================="
    print ">>> random by sampling in the {0, 1} distribution of y is at accuracy:",
    tot = y.sum() * 1. / y.shape[0]
    print max(1-tot, tot)

    print "================="
    print ">>> as a bag of features without any structure between electrodes nor timesteps"
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)
    train_models(X_train, y_train, X_test, y_test)

    tmp = d['X'].shape
    X = d['X'].sum(axis=2)
    y = d['y'].ravel()
    print X.shape
    print y.shape

    print "================="
    print ">>> now we're just summing all electrodes along the time axis"
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)
    train_models(X_train, y_train, X_test, y_test)

