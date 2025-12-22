
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

def get_model(method, alpha=1.0, l1_ratio=0.5):
    if method == "OLS":
        return LinearRegression()
    if method == "Ridge":
        return Ridge(alpha=alpha)
    if method == "Lasso":
        return Lasso(alpha=alpha)
    if method == "Elastic Net":
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
