# gradient descent 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import clear_output

# 날씨를 예측하는 perceptron 구현
# 아침에 구름이 적고 바랍이 약하면 => 맑은 날 확률 up
# 아침에 구름이 많고 바람이 강하면 => 비올 확률 up
# 구름과 바람이 X1, X2


# 계단 함수 정의
# 역치보다 크면 1 아니면 0; 여기서의 역치는 0.5로 지정
def step_function(x) :
    if x < thres:
        return 0
    else :
        return 1

# 랜덤으로 데이터를 만들어주는 function
# 실제 데이터
def gen_training_data(data_point):
    # 0 ~ 1까지 랜덤으로 생성해줌
    x1 = np.random.random(data_point)
    x2 = np.random.random(data_point)
    # x1 + x2가 1보다 크면 1 아니면 0
    y = ((x1 + x2) > 1).astype(int)
    # 하나의 데이터이고 이것을 100개로 만들 예정
    training_set = [((x1[i], x2[i]), y[i]) for i in range(len(x1))]

    return training_set
# 역치
thres = 0.5
w = np.array([0.3, 0.9])
# learing rate
lr = 0.1
data_point = 100
# 데이터를 얼마나 반복적으로 할 것인가
epoch = 10
training_set = gen_training_data(data_point)
# print(training_set[0:5])
# figure 그리기
plt.figure(0)
plt.ylim(-0.1, 1.1)
plt.xlim(-0.1, 1.1)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

for x, y in training_set:
    if y == 1:
        plt.plot(x[0],x[1], 'bo')
    else:
        plt.plot(x[0],x[1], 'go')
plt.show()

# 애니메이션 처럼 보이기 위한 것
# %matplotlib inline
from time import sleep
plt.figure(0)
plt.ylim(-0.1, 1.1)
plt.xlim(-0.1, 1.1)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

xx = np.linspace(0,1,50)
for i in range(epoch):
    cnt = 0
    for x , y in training_set:
        clear_output(wait=True)
        # 입력값 * 연결 강도
        u = sum(x * w)
        error = y - step_function(u)
        for index, value in enumerate(x):
            # perceptron 학습방법
            # 새 연결 강도 = 현 연결강도 + 현 입력값 * 오차 * 학습률
            w[index] = w[index] + lr * error * value
        for xs, ys in training_set[0:cnt]:
            plt.ylim(-0.1,1.1)
            plt.xlim(-0.1,1.1)
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')

            if ys == 1:
                plt.plot(xs[0], xs[1], 'bo')
            else : 
                plt.plot(xs[0], xs[1], 'go')
        
        yy = -w[1]/w[0] * xx + thres /w[0] # <== w[0] * yy + w[1] * xx = thres
        # plot 그리기
        plt.plot(xx, yy)
        plt.show()
        cnt = cnt+ 1
        sleep(0.01)
