'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np


#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree=1, reg_lambda=1E-8):
        """
        Constructor
        """
        self.degree = degree
        self.regLambda = reg_lambda
        self.theta = None
        self.Xmean = None
        self.Xstd = None

    def polyfeatures(self, X, degree):
        """
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not include the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        """
        X_ = X.flatten()
        degree_ = np.arange(1, degree+1) # array of polynomial exponents 1 -> degree
        return np.power.outer(X_, degree_) # return polynomial features

    def fit(self, X, y):
        """
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        """
        n = len(X)
        
        # matrix of polynomial features
        X_ = self.polyfeatures(X, self.degree)
        # standardize features
        self.Xmean = X_.mean(axis=0)
        self.Xstd = X_.std(axis=0)
        X_ = (X_ - self.Xmean) / self.Xstd
        # add column of ones to front, for x^0
        X_ = np.c_[np.ones(n), X_]

        # regularization matrix
        regMatrix = self.regLambda * np.identity(self.degree + 1)
        regMatrix[0, 0] = 0

        # analytic solution (X'X + regMatrix)^-1 X' y
        self.theta = np.linalg.pinv(X_.T.dot(X_) + regMatrix).dot(X_.T).dot(y)

    def predict(self, X):
        """
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        """
        n = len(X)

        # matrix of polynomial features
        X_ = self.polyfeatures(X, self.degree)
        # standardize features
        X_ = (X_ - self.Xmean) / self.Xstd
        # add column of ones to front, for x^0
        X_ = np.c_[np.ones(n), X_]

        return X_ @ self.theta


#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------



def learningCurve(Xtrain, Ytrain, Xtest, Ytest, reg_lambda, degree):
    """
    Compute learning curve

    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree

    Returns:
        errorTrain -- errorTrain[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTest -- errorTrain[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]

    Note:
        errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """

    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)

    for i in range(1,n):

        # fit on the first i training points
        model = PolynomialRegression(degree=degree, reg_lambda=reg_lambda)
        model.fit(Xtrain[:i+1], Ytrain[:i+1])

        # predict on first i training points and compute error
        Ytrain_pred = model.predict(Xtrain[:i+1])
        errorTrain[i] = np.mean((Ytrain[:i+1] - Ytrain_pred)**2)

        # predict on whole test set and compute error
        Ytest_pred = model.predict(Xtest)
        errorTest[i] = np.mean((Ytest - Ytest_pred)**2)

    return errorTrain, errorTest
