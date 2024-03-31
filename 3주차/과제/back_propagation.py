# gradient desent 
import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(100)
# print(x)
Y = 0.2 * X * 0.5
plt.figure(figsize=(8,6))
plt.scatter(X,Y)
plt.show()


def plot_prediction(pred,y) : 
    plt.figure(figsize=(8,6))
    plt.scatter(X, y)
    plt.scatter(X,pred)
    plt.show()

## gradient Descent 구현
# 업데이트 할 W : Learning Rate * ((Y예측 - Y실제) * X) 평균
# 업데이트 할 b : Learning Rate * ((Y예측 - Y실제) * 1) 평균
W = np.random.uniform(-1, 1)
b = np.random.uniform(-1, 1)

learning_rate = 0.7

for epoch in range(200):
    Y_Pred = W * X + b

    error = np.abs(Y_Pred - Y).mean()
    if error < 0.001:
        break

    # gradient descent 계산
    w_grad = learning_rate * ((Y_Pred - Y) * X).mean()
    b_grad = learning_rate * ((Y_Pred - Y) * 1).mean()

    # W, b 값 갱신
    W = W - w_grad
    b = b - b_grad

    if epoch % 10 == 0:
        Y_Pred = W * X + b
        plot_prediction(Y_Pred, Y)