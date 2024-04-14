#backpropagation
import numpy as np

class MLP:
    def  __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 가중치 초기화
        # self.w1_2_3_4 = np.random.random((self.input_size, self.hidden_size))
        self.w1_2_3_4 = [[1,10],[1,10]]

        # self.w5_6 =  np.random.random((self.hidden_size, self.output_size))
        self.w5_6 =  [[-40],[40]]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):

        self.z1_2 = np.dot(x, self.w1_2_3_4)
        