# Ricardo Corral Corral - abril, 2015

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from collections import defaultdict
from scipy.linalg import sqrtm, inv
import grand_schmidt
from random import uniform
from sklearn import cross_validation
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.manifold import Isomap, TSNE, SpectralEmbedding, MDS
from sklearn import preprocessing
from sklearn.linear_model import MultiTaskLasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR, SVC, LinearSVC
from sklearn.preprocessing import normalize
from numpy.linalg import norm
from numpy import inner



def scalarProjection(a,b):
    return inner(a,b)/norm(b)

class classSparser(object):
    def __init__(self,mapperType='PIMP',support=150,projectOnSubspace=False):
        #options are
        #'PIMP' for Moore Penrose Pseudo Inverse
        #'Regressor' for using a regression task on each dimension
        self.mapperType = mapperType
        self.sparsed_X = None
        self.transformation_matrix = None
        self.Regressor = None
        self.support = support
        self.projectOnSubspace = projectOnSubspace

    def fit(self,X,Y):
        self.sparsed_X = list()
        #First, tranlate points to the origin
        main_centroid = [ np.mean(x) for x in np.transpose(X) ]
        print 'Main centroid:', main_centroid
        X = X - main_centroid

        byClassDict = defaultdict(list)
        for i in xrange(len(Y)):
            byClassDict[Y[i]].append(X[i])


        class_centroids = dict()

        centroids_matrix = list()
        kindexmap = dict()

        _i = 0
        for k in byClassDict:
            class_centroid = [ np.mean(x) for x in np.transpose(byClassDict[k]) ] #np.mean(byClassDict[k])
            _norm = np.linalg.norm(class_centroid)
            _scaling_factor = _norm**2#(i+1)**2 #+ (i+_norm)  #Play with this using _norm, i and any otrher function/constant
            _centroid = np.array(class_centroid)#*(_scaling_factor)
            print '*** Class centroid:', _centroid
            class_centroids[k] = _centroid
            centroids_matrix.append(_centroid)
            kindexmap[k] = _i
            _i+=1

        centroids_matrix = np.array(centroids_matrix)
        ortho_centroids_matrix = np.array(grand_schmidt.gs(centroids_matrix))
        ortho_centroids_matrix = normalize(ortho_centroids_matrix)

        print '*Centroids matrix',centroids_matrix
        print '*Ortho centroids matrix', ortho_centroids_matrix


        newX, newY = list(), list()
        ks = list()
        for k in byClassDict:
            #byClassDict[k] = np.array(byClassDict[k]) - centroids_matrix[kindexmap[k]] + np.array(ortho_centroids_matrix[kindexmap[k]]) #class_centroids[k]

            #this is the basis vector corresponding to current class
            classvector = np.array(ortho_centroids_matrix[kindexmap[k]])
            kScalingFactor = self.support

            #This section tries to get a good scaling factor for each orthonormal vector
            maxks = list()
            for _k in ks:
                projs = [scalarProjection(x,classvector) for x in byClassDict[_k]]
                maxk = max(projs)
                maxks.append(maxk)

                maxownk = max([scalarProjection(x,classvector) for x in byClassDict[k]])

            if len(ks):
                kScalingFactor = max(maxks) + abs(maxownk) + self.support


            for v in byClassDict[k]:
                vv = np.array(v) - centroids_matrix[kindexmap[k]] + classvector*kScalingFactor
                self.sparsed_X.append(vv)
                newX.append(v)
                newY.append(k)
                ks.append(k)

        self.sparsed_X = np.array(self.sparsed_X)

        if self.projectOnSubspace:
            #Project on to new subspace spawned by class vectors
            self.sparsed_X = np.dot(self.sparsed_X,np.transpose(centroids_matrix) )


        if self.mapperType == 'PIMP':
            #self.scaler = preprocessing.StandardScaler().fit(self.sparsed_X)
            #self.sparsed_X = self.scaler.transform(self.sparsed_X)

            self.transformation_matrix = self.sparsed_X*(np.transpose(np.linalg.pinv(X) ) )
            #self.transformation_matrix = X*(np.transpose(np.linalg.pinv(self.sparsed_X) ) )

        if self.mapperType == 'Regressor':
            self.Regressor = MultiTaskLasso(alpha=0.00000001,max_iter=2000)
            self.Regressor.fit(newX,self.sparsed_X)

        return self.sparsed_X, newY


    def transform(self,X):
        Xs = X#self.scaler.transform(X)
        if self.mapperType == 'PIMP':
            transformed_data = self.transformation_matrix*Xs
            #transformed_data = Xs*self.transformation_matrix
        if self.mapperType == 'Regressor':
            transformed_data = self.Regressor.predict(Xs)

        return transformed_data

def main():
    _dataset = datasets.load_iris()
    #_dataset = datasets.load_digits()
    X = _dataset.data
    Y = _dataset.target

    cS = classSparser(mapperType='Regressor')
    X2, Y2 = cS.fit(X,Y)
    X3 = cS.transform(X)



    for i in xrange(2):
        X2, Y2 = cS.fit(X3,Y)
        X3 = cS.transform(X3)

    X_a = MDS(n_components=2).fit_transform(X)
    X_b = MDS(n_components=2).fit_transform(X2)
    X_c = MDS(n_components=2).fit_transform(X3)


    fig = plt.figure()
    a=fig.add_subplot(1,3,1)
    plt.title("original")
    plt.scatter(X_a[:, 0], X_a[:, 1], c=Y, cmap=plt.cm.Paired)

    a=fig.add_subplot(1,3,2)
    plt.title("transformed")
    plt.scatter(X_b[:, 0], X_b[:, 1], c=Y2, cmap=plt.cm.Paired)

    a=fig.add_subplot(1,3,3)
    plt.title("approximated")
    plt.scatter(X_c[:, 0], X_c[:, 1], c=Y, cmap=plt.cm.Paired)


    clf = GaussianNB() #LinearSVC()
    scoresA = np.mean(cross_validation.cross_val_score(clf, X,  Y, cv=5) )
    scoresB = np.mean(cross_validation.cross_val_score(clf, X2, Y2, cv=5) )
    scoresC = np.mean(cross_validation.cross_val_score(clf, X3, Y, cv=5) )

    print 'Scores:', scoresA, scoresB, scoresC


    plt.show()

if __name__ == '__main__':
    main()
