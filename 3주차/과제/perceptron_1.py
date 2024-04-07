# gradient descent 
import numpy as np
import matplotlib.pyplot as plt


X = np.random.rand(100)
# X2 = np.random.rand(100)
print(X)
# 임의의 f(x) 정의
# y = wx + b <= 1차 방정식
Y = 0.2 * X + 0.5
plt.figure(figsize=(8,6))
plt.scatter(X,Y)
plt.show()


def plot_prediction(pred,y) : 
    plt.figure(figsize=(8,6))
# 실제값을 보여줌
    plt.scatter(X, y)
# 예측 값을 보여줌
    plt.scatter(X,pred)
    plt.show()

## gradient Descent 구현
# 업데이트 할 W : Learning Rate * ((Y예측 - Y실제) * X) 평균
# 업데이트 할 b : Learning Rate * ((Y예측 - Y실제) * 1) 평균
#.  -1 ~ 1까지  랜덤 수
W = np.random.uniform(-1, 1)
# W2 = np.random.uniform(-1, 1)
b = np.random.uniform(-1, 1)
print(W) 
print(b)
learning_rate = 0.7

for epoch in range(200):
    Y_Pred = W * X + b

# 예측값 - 실제값의 절대값의 평균
    error = np.abs(Y_Pred - Y).mean()
    # 에러 잡는 이유 : 에러가 어느 정도 도달하면 멈추기 위해서
    if error < 0.001:
        break

    # gradient descent 계산
    w_grad = learning_rate * ((Y_Pred - Y) * X).mean()
    # w_grad_2 = learning_rate * ((Y_Pred - Y) * X2).mean()
    b_grad = learning_rate * ((Y_Pred - Y) * 1).mean()

    # W, b 값 갱신
    # !질문 : 왜 다시 갱신을 해줄까
    W = W - w_grad
    # W = W2 - w_grad_2
    b = b - b_grad

    if epoch % 10 == 0:
        Y_Pred = W * X + b
        plot_prediction(Y_Pred, Y)

# 최소 제곱법
