import numpy as np

# 0부터 3까지 배열 만들기
arr1 = np.arange(4)
print(arr1)

arr2 = np.zeros((4,4), dtype= float)
print(arr2)

arr3 = np.ones((3,3), dtype= str)
print(arr3)

# 0부터 9까지 랜덤하게 초기화된 배열 만들기
arr4 = np.random.randint(0,10,(3,3))
print(arr4)

#평균이 0이고, 표준편차가 1인 표준 정규를 띄운 배열
arr5 = np.random.normal(0,1,(3,3))
print(arr5)