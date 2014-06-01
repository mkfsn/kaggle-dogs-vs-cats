import os
import numpy as np
from scipy.stats import sem
from nolearn.convnet import ConvNetFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, classification_report

DECAF_IMAGENET_DIR = 'imagenet_pretrained/'
TRAIN_DATA_DIR = 'images/'

def get_dataset():
    cat_dir = TRAIN_DATA_DIR + 'cat/'
    cat_filenames = [cat_dir + fn for fn in os.listdir(cat_dir)]
    dog_dir = TRAIN_DATA_DIR + 'dog/'
    dog_filenames = [dog_dir + fn for fn in os.listdir(dog_dir)]

    labels = [0] * len(cat_filenames) + [1] * len(dog_filenames)
    filenames = cat_filenames + dog_filenames
    return shuffle(filenames, labels, random_state=0)

def evaluate_cross_validation(_clf, _X, _y, _K=5):
    _cv = KFold(len(_y), _K, shuffle=True, random_state=0)
    _scores = cross_val_score(_clf, _X, _y, cv=_cv)
    print 'scores:',_scores
    print ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(_scores), sem(_scores))
    print

def train_and_evaluate(_clf, _X_train, _X_test, _y_train, _y_test):
    _clf.fit(_X_train, _y_train)

    print 
    print "*" * 50
    print "Accuracy on Training Set:"
    print '{0:.3f}'.format(_clf.score(_X_train, _y_train))
    print
    print "Accuracy on Testing Set:"
    print '{0:.3f}'.format(_clf.score(_X_test, _y_test))
    print

    _y_pred = _clf.predict(_X_test)

    print "Classification Report on Testing Set:"
    print classification_report(_y_test, _y_pred)
    print
    print "Confusion Matrix on Testing Set:"
    print confusion_matrix(_y_test, _y_pred)
    print "*" * 50

def main():
    convnet = ConvNetFeatures(
        pretrained_params=DECAF_IMAGENET_DIR + 'imagenet.decafnet.epoch90',
        pretrained_meta=DECAF_IMAGENET_DIR + 'imagenet.decafnet.meta',
        )
    clf = LogisticRegression()
    pl = Pipeline([('convnet', convnet), ('clf', clf)])

    X, y = get_dataset()
    X, y = X[:300], y[:300]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.67, random_state=0)

    # K-Fold Cross-Validation (default 5-fold)
    # evaluate_cross_validation(pl, X, y, 5)
    
    # Accuracy, Classification Report, and Confusion Matrix
    train_and_evaluate(pl, X_train, X_test, y_train, y_test)

    # Accuracy on Testing Set
    # print "Fitting..."
    # pl.fit(X_train, y_train)
    # print "Predicting..."
    # y_pred = pl.predict(X_test)
    # print "Accuracy: %.3f" % accuracy_score(y_test, y_pred)

if __name__ == "__main__":
    main()
