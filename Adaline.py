import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入数据
df = pd.read_csv("./iris.data", header=None)

y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", -1, 1)

X = df.iloc[0:100, [0, 2]].values


class AdalineGD:

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) > 0.0, 1, -1)

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1] + 1)
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ad1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ad1.cost_)+1), np.log10(ad1.cost_), marker='o')
ax[0].set_xlabel("epoches")
ax[0].set_ylabel("sum-square-error")
ax[0].set_title("learning rate 0.01")
ad2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ad2.cost_)+1), ad2.cost_, marker='o')
ax[1].set_xlabel("epoches")
ax[1].set_ylabel("sum-square-error")
ax[1].set_title("learning rate 0.0001")
plt.show()

# 特征缩放
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0]-X[:, 0].mean())/X[:, 0].std()
X_std[:, 1] = (X[:, 1]-X[:, 1].mean())/X[:, 1].std()
ad3 = AdalineGD(n_iter=15, eta=0.01)
ad3.fit(X_std, y)
plt.plot(range(1, len(ad3.cost_)+1), ad3.cost_, marker="o")
plt.show()
