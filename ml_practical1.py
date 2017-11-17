import _pickle as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

X,y=cp.load(open('winequality-white.pickle','rb'))

N,D=X.shape
N_train = int(0.8*N)
N_test = N - N_train

X_train = X[:N_train]
y_train = y[:N_train]
X_test = X[N_train:]
y_test = y[N_train:]

#print(y_test)
#handin 1 plot bar chart
print("handin 1 plot bar chart")
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
print("handin2 baseline")
predict = np.average(y_train)
print(predict)

y_predict_trivial = [predict]*len(y_test)
#print(y_predict_trivial)
squared_err_sum = 0
err = y_predict_trivial - y_test
mean_squared_err_trivial = np.sum(err*err)/len(y_test)
print(mean_squared_err_trivial)

#handin 3 Linear Model Using LSE
print("handin 3 linear model using LSE")
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
#print(np.shape(new_X_train))
w = np.dot(np.linalg.inv(np.dot(new_X_train.T,new_X_train)),new_X_train.T)
#print(np.shape(w))
#print(np.shape(y_train))
w = np.dot(w,y_train)


#train error
y_train_LSE = np.dot(new_X_train,w)
#print(y_predict_LSE)
#print(y_predict_LSE)
squared_err_sum = 0
err = y_train_LSE - y_train
mean_squared_err_LSE = np.sum(err*err)/len(y_train)
#print(err)
print("train error")
print(mean_squared_err_LSE)

#predict
w_0 = np.ones((len(normalized_X_test[:,0]),1))
new_X_test = np.hstack((w_0,normalized_X_test))
y_predict_LSE = np.dot(new_X_test,w)
#print(y_predict_LSE)
#print(y_predict_LSE)
squared_err_sum = 0
err = y_predict_LSE - y_test
mean_squared_err_LSE = np.sum(err*err)/len(y_test)
#print(err)
print("test error")
print(mean_squared_err_LSE)


#handin 4 
print("learning curves plot")
result_error_train = [0]*20
result_error_test = [0]*20
for k in range(0,20):
    train_size = (k+1)*20
    train_data = X_train[0:train_size,:]
    train_label = y_train[0:train_size]
    train_avg = [0,0,0,0,0,0,0,0,0,0,0]
    train_std = [0,0,0,0,0,0,0,0,0,0,0]
    for i in range(0,11):
        train_avg[i] = np.mean(train_data[:,i])
        train_std[i] = np.std(train_data[:,i])
    test = 0
    #for i in range(0,len(X_train[:,1])):
    #    test = test + X_train[i,1]
    #print(test)
    #print(train_avg)
    normalized_X_train = train_data.copy()
    for i in range(0,11):
        for j in range(0,len(train_data[:,i])):
            normalized_X_train[j,i] = (train_data[j,i]-train_avg[i])/train_std[i]

    #compute w
    #print(np.shape(normalized_X_train.T))
    w_0 = np.ones((len(normalized_X_train[:,0]),1))
    new_X_train = np.hstack((w_0,normalized_X_train))
    #print(np.shape(new_X_train))
    w = np.dot(np.linalg.inv(np.dot(new_X_train.T,new_X_train)),new_X_train.T)
    #print(np.shape(w))
    #print(np.shape(y_train))
    w = np.dot(w,train_label)


    #train error
    y_train_LSE = np.dot(new_X_train,w)
    #print(y_predict_LSE)
    #print(y_predict_LSE)
    squared_err_sum = 0
    err = y_train_LSE - train_label
    mean_squared_err_LSE = np.sum(err*err)/len(train_label)
    #print(err)
    print("train error")
    print(mean_squared_err_LSE)
    result_error_train[k] = mean_squared_err_LSE
    #predict
    w_0 = np.ones((len(normalized_X_test[:,0]),1))
    new_X_test = np.hstack((w_0,normalized_X_test))
    y_predict_LSE = np.dot(new_X_test,w)
    #print(y_predict_LSE)
    #print(y_predict_LSE)
    squared_err_sum = 0
    err = y_predict_LSE - y_test
    mean_squared_err_LSE = np.sum(err*err)/len(y_test)
    #print(err)
    print("test error")
    print(mean_squared_err_LSE)
    result_error_test[k] = mean_squared_err_LSE

#print(result_error_train)
xlabel = [0]*20
for i in range(0,20):
    xlabel[i] = i+1;
fig,ax = plt.subplots()
ax.plot(result_error_train)
ax.plot(result_error_test)
fig.tight_layout()
plt.show()

#optional Polynomial Basis Expansion with Ridge and Lasso
print("optional Polynomial Basis Expansion with Ridge and Lasso")
valid = int(0.8*N_train)
train_data = X_train[:valid,:]
train_label = y_train[:valid]
valid_data = X_train[valid:,:]
valid_label = y_train[valid:]
a = [0.01,0.1,1,10,100]

for i in range(0,5):
    #lasso
    model = make_pipeline(PolynomialFeatures(2),StandardScaler(),linear_model.Lasso(alpha=a[i]))
    model.fit(train_data,train_label)
    valid_predict = model.predict(valid_data)
    err = valid_predict-valid_label
    mean_squared_err_LSE = np.sum(err*err)/len(valid_label)
    #print(err)
    #print("valid error")
    #print(mean_squared_err_LSE)

min_err_lasso_alpha = 0.01
model = make_pipeline(PolynomialFeatures(2),StandardScaler(),linear_model.Lasso(alpha=0.01))
model.fit(X_train,y_train)
valid_predict = model.predict(X_train)
err = valid_predict-y_train
mean_squared_err_LSE = np.sum(err*err)/len(y_train)
print("lasso train error")
print(mean_squared_err_LSE)
valid_predict = model.predict(X_test)
err = valid_predict-y_test
mean_squared_err_LSE = np.sum(err*err)/len(y_test)
print("lasso test error")
print(mean_squared_err_LSE)


for i in range(0,5):
    #Ridge
    model = make_pipeline(PolynomialFeatures(2),StandardScaler(),linear_model.Ridge(alpha=a[i]))
    model.fit(train_data,train_label)
    valid_predict = model.predict(valid_data)
    err = valid_predict-valid_label
    mean_squared_err_LSE = np.sum(err*err)/len(valid_label)
    #print(err)
    print("valid error")
    print(mean_squared_err_LSE)

min_err_ridge_alpha = 0.01
model = make_pipeline(PolynomialFeatures(2),StandardScaler(),linear_model.Ridge(alpha=0.01))
model.fit(X_train,y_train)
valid_predict = model.predict(X_train)
err = valid_predict-y_train
mean_squared_err_LSE = np.sum(err*err)/len(y_train)
print("ridge train error")
print(mean_squared_err_LSE)
valid_predict = model.predict(X_test)
err = valid_predict-y_test
mean_squared_err_LSE = np.sum(err*err)/len(y_test)
print("ridge test error")
print(mean_squared_err_LSE)

#Super-optional Larger Degrees
print("Super-optional Larger Degrees")
#for lasso
for i in range(0,5):
    skf = StratifiedKFold(n_splits=3,shuffle=True)
    skf.get_n_splits(X_train, y_train)
    avg_err = 0;
    model = make_pipeline(PolynomialFeatures(3),StandardScaler(),linear_model.Lasso(alpha=a[i]))
    for train_index, test_index in skf.split(X_train, y_train):
        X_train_fold, X_valid = X_train[train_index], X_train[test_index]
        y_train_fold, y_valid = y_train[train_index], y_train[test_index]
        model.fit(X_train_fold,y_train_fold)
        valid_predict = model.predict(X_valid)
        err = valid_predict-y_valid
        mean_squared_err_LSE = np.sum(err*err)/len(y_valid)
        avg_err = avg_err + mean_squared_err_LSE
    avg_err = avg_err/3
    print(avg_err)
    
min_err_lasso_alpha = 0.01
#for Ridge
for i in range(0,5):
    skf = StratifiedKFold(n_splits=3,shuffle=True)
    skf.get_n_splits(X_train, y_train)
    avg_err = 0;
    model = make_pipeline(PolynomialFeatures(3),StandardScaler(),linear_model.Ridge(alpha=a[i]))
    for train_index, test_index in skf.split(X_train, y_train):
        X_train_fold, X_valid = X_train[train_index], X_train[test_index]
        y_train_fold, y_valid = y_train[train_index], y_train[test_index]
        model.fit(X_train_fold,y_train_fold)
        valid_predict = model.predict(X_valid)
        err = valid_predict-y_valid
        mean_squared_err_LSE = np.sum(err*err)/len(y_valid)
        avg_err = avg_err + mean_squared_err_LSE
    avg_err = avg_err/3
    print(avg_err)

min_err_ridge_alpha = 100