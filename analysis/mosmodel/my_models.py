from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, LassoLarsCV
from sklearn.preprocessing import MaxAbsScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

def getPolyModel(degree):
    poly_model = Pipeline([
        ('scale', MaxAbsScaler()),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('linear', LinearRegression(fit_intercept=True))])
    return poly_model
poly1 = getPolyModel(1)
poly2 = getPolyModel(2)
poly3 = getPolyModel(3)

def getMosmodel(degree):
    mosmodel = Pipeline([
        ('scale', MaxAbsScaler()),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('linear', LassoLarsCV(fit_intercept=True))])
        #('linear', LassoLarsCV(fit_intercept=True, cv=5, eps=1e-6))])
    return mosmodel

mosmodel = getMosmodel(3)

from . import utility
def calculateModelError(model, train_df, test_df, features):
    x_test = test_df[features]
    y_true = test_df['cpu-cycles']
    y_pred = predictRuntime(model, train_df, features, x_test)
    error = utility.relativeError(y_true, y_pred)
    return error

def predictRuntime(model, train_df, features, x):
    x_train = train_df[features]
    y_train = train_df['cpu-cycles']
    model.fit(x_train, y_train)

    return model.predict(x)

