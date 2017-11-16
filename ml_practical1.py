import _pickle as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

X,y=cp.load(open('winequality-white.pickle','rb'))

N,D=X.shape
N_train = int(0.8*N)
N_test = N - N_train

X_train = X[:N_train]
y_train = y[:N_train]
X_test = X[N_train:]
y_test = y[N_train:]

print(y_test)
#handin 1 plot bar chart

n_groups = 7

fig,ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.4
y_num = [0,0,0,0,0,0,0]
for i in range(0,7):
    for score in y_train:
        if score == i+3:
            y_num[i]=y_num[i] + 1
#print(y_num)
rect = ax.bar(index + bar_width, y_num, bar_width, alpha = opacity, color = 'r', label = 'y')
ax.set_xlabel('Score')
ax.set_ylabel('Number')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('3','4','5','6','7','8','9'))

fig.tight_layout()
plt.show()

#handin 2 baseline
predict = np.average(y_train)
#print(predict)
y_predict_trivial = np.arange(len(y_test))
for i in range(0,len(y_test)):
    y_predict_trivial[i] = predict
#print(y_predict_trivial)
squared_err_sum = 0
err = y_predict_trivial - y_test
for i in err:
    squared_err_sum = squared_err_sum + i*i
mean_squared_err_trivial = squared_err_sum/len(y_test)
print(mean_squared_err_trivial)

#handin 3 Linear Model Using LSE
#print(np.shape(y_train))
#print(np.shape(X_train))
train_avg = [0,0,0,0,0,0,0,0,0,0,0]
train_std = [0,0,0,0,0,0,0,0,0,0,0]
for i in range(0,11):
    train_avg[i] = np.mean(X_train[:,i])
    train_std[i] = np.std(X_train[:,i])
test = 0
#for i in range(0,len(X_train[:,1])):
#    test = test + X_train[i,1]
#print(test)
#print(train_avg)
normalized_X_train = X_train.copy()
for i in range(0,11):
    for j in range(0,len(X_train[:,i])):
        normalized_X_train[j,i] = (X_train[j,i]-train_avg[i])/train_std[i]

#test the normalization result
'''
ttrain_avg = [0,0,0,0,0,0,0,0,0,0,0]
ttrain_std = [0,0,0,0,0,0,0,0,0,0,0]
for i in range(0,11):
    ttrain_avg[i] = np.mean(normalized_X_train[:,i])
    ttrain_std[i] = np.std(normalized_X_train[:,i])
print(ttrain_avg)
print(ttrain_std)
'''

#normalize test data
normalized_X_test = X_test.copy()
for i in range(0,11):
    for j in range(0,len(X_test[:,i])):
        normalized_X_test[j,i] = (X_test[j,i]-train_avg[i])/train_std[i]      
        

#compute w
#print(np.shape(normalized_X_train.T))
w_0 = np.ones((len(normalized_X_train[:,0]),1))
new_X_train = np.hstack((w_0,normalized_X_train))
print(np.shape(new_X_train))
w = np.dot(np.linalg.inv(np.dot(new_X_train.T,new_X_train)),new_X_train.T)
print(np.shape(w))
print(np.shape(y_train))
w = np.dot(w,y_train)

#predict
w_0 = np.ones((len(normalized_X_test[:,0]),1))
new_X_test = np.hstack((w_0,normalized_X_test))
y_predict_LSE = np.dot(new_X_test,w)
print(y_predict_LSE)
#print(y_predict_LSE)
squared_err_sum = 0
err = y_predict_LSE - y_test
for i in err:
    squared_err_sum = squared_err_sum + i*i
mean_squared_err_LSE = squared_err_sum/len(y_test)
print(err)
print(mean_squared_err_LSE)

#handin 4 











