# -*- coding: utf-8 -*-


import numpy as np

import matplotlib.pyplot as plt

'''产生数据[N*K,D]'''
# 每个类中的样本点
N = 100 

# 维度
D = 2

# 类别个数
K = 4 

# 样本input
X = np.zeros((N*K,D)) 

# 类别标签
y = np.zeros(N*K, dtype='uint8') 


for j in xrange(K):

  ix = range(N*j,N*(j+1))

  r = np.linspace(0.0,1,N) # radius

  t = np.linspace(j*5,(j+1)*5,N) + np.random.randn(N)*0.2 # theta

  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]

  y[ix] = j

'''单隐层神经网络 '''
# 随机初始化参数

# 隐层大小
h = 200 

# 第一层：W1权重（D*h） b1偏移（h个）
W1 = 0.01 * np.random.randn(D,h)
b1 = np.zeros((1,h))

# 第二层：W2权重（h*K） b2偏移（K个）
W2 = 0.01 * np.random.randn(h,K)
b2 = np.zeros((1,K))

#步长
step_size = 1e-0

#正则化系数
reg = 1e-3

model = {}

#点的个数(300)
num_examples = X.shape[0]

# 梯度迭代与循环
for i in xrange(12000):

  '''计算类别得分'''
  #2层神经网络的前向计算
  hidden_layer = np.maximum(0, np.dot(X, W1) + b1) #使用的ReLU神经元
  scores = np.dot(hidden_layer, W2) + b2

  '''计算类别概率（Softmax函数）'''
  # 用指数函数还原
  exp_scores = np.exp(scores) 
  # 归一化
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

  '''计算损失loss(包括互熵损失和正则化部分)'''
  #互熵损失
  corect_logprobs = -np.log(probs[range(num_examples),y])
  data_loss = np.sum(corect_logprobs)/num_examples

  #正则化部分
  reg_loss = 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)

  #损失
  loss = data_loss + reg_loss

  if i % 100 == 0:
    print "iteration %d: loss %f" % (i, loss)

  # 计算得分上的梯度
  dscores = probs
  dscores[range(num_examples),y] -= 1
  dscores /= num_examples

  # 梯度回传
  dW2 = np.dot(hidden_layer.T, dscores)
  db2 = np.sum(dscores, axis=0, keepdims=True)

  dhidden = np.dot(dscores, W2.T)

  dhidden[hidden_layer <= 0] = 0

  # 拿到最后W,b上的梯度
  dW1 = np.dot(X.T, dhidden)
  db1 = np.sum(dhidden, axis=0, keepdims=True)

  # 加上正则化梯度部分
  dW2 += reg * W2
  dW1 += reg * W1

  # 参数迭代与更新
  W1 += -step_size * dW1
  b1 += -step_size * db1
  W2 += -step_size * dW2
  b2 += -step_size * db2

  model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

#计算分类准确度
hidden_layer = np.maximum(0, np.dot(X, W1) + b1)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print 'training accuracy: %.2f' % (np.mean(predicted_class == y))

# 可视化样本点
# Helper function to predict an output (0 or 1)
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
  
    a1 = np.maximum(0, np.dot(x, W1) + b1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


plot_decision_boundary(lambda x: predict(model, x))

plt.show()
