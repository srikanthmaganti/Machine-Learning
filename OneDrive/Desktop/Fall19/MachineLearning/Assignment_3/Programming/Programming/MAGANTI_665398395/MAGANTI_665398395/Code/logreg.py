import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import math

def predict(X, w):
    n_ts = X.shape[0]
    n_vars = X_tr.shape[1]  # number of variables
    # use w for prediction
    pred = np.zeros(n_ts)       # initialize prediction vector
    for i in range(n_ts):
        # TODO
        # Compute temp = exp(w*x_i) / (1 + exp(w*x_i))
        y=np.dot(w.T,X[i])
        pred[i]= 1 if y>0 else 0

                        # compute your prediction
    return pred

def accuracy(X, y, w):
    y_pred,count = predict(X, w),0
    # TODO
    accuracy=(float(sum(y_pred==y)/len(y)))*100
    return accuracy


def logistic_reg(X_tr, X_ts, lr, Total_Iterations):
    #perform gradient descent
    test_accuracy,train_accuracy=[],[]
    n_vars = X_tr.shape[1]  # number of variables
    n_tr = X_tr.shape[0]    # number of training examples
    w = np.zeros(n_vars)    # initialize parameter w
    tolerance = 0.01        # tolerance for stopping

    iter = 0                # iteration counter
    max_iter = 1001         # maximum iteration

    while (True):
        iter += 1

        # calculate gradient
        grad = np.zeros(n_vars) # initialize gradient
        for i in range(n_tr):
            
            # Compute temp = exp(w*x_i) / (1 + exp(w*x_i))
            weight_values=np.dot(w.T,X_tr[i])
            if weight_values>709:
            	weight_values=709

            temp=(np.exp(weight_values))/(1+np.exp(weight_values))


            for j in range(n_vars):
                # TODO
                # grad[j] = grad[j] + (y_i - temp) x_{ij}
                grad[j]=grad[j] + (y_tr[i]-temp)*(X_tr[i][j])

        # TODO
        # w_new = w + lr * grad
       
        w_new = w + (lr*grad)
       


        if iter%50 == 0:
            print('Iteration: {0}, mean abs gradient = {1}'.format(iter, np.mean(np.abs(grad))))

        # stopping criteria and perform update if not stopping
        if (np.mean(np.abs(grad)) < tolerance):
            w = w_new
            break
        else :
            w = w_new

        if (iter >= max_iter):
            break
        if iter in Total_Iterations:
        	test_accuracy.append(accuracy(X_ts,y_ts,w))
        	train_accuracy.append(accuracy(X_tr,y_tr,w))

    #test_accuracy  = accuracy(X_ts, y_ts, w)
    #train_accuracy = accuracy(X_tr, y_tr, w)
    return test_accuracy, train_accuracy
def logistic_reg_regularized(X_tr, X_ts, lr, lambda_value):
    #perform gradient descent
    #test_accuracy,train_accuracy=[],[]
    n_vars = X_tr.shape[1]  # number of variables
    n_tr = X_tr.shape[0]    # number of training examples
    w = np.zeros(n_vars)    # initialize parameter w
    tolerance = 0.01        # tolerance for stopping

    iter = 0                # iteration counter
    max_iter = 1001         # maximum iteration

    while (True):
        iter += 1

        # calculate gradient
        grad = np.zeros(n_vars) # initialize gradient
        for i in range(n_tr):
            
            # Compute temp = exp(w*x_i) / (1 + exp(w*x_i))
            weight_values=np.dot(w.T,X_tr[i])
            if weight_values>709:
            	weight_values=709

            temp=(np.exp(weight_values))/(1+np.exp(weight_values))


            for j in range(n_vars):
                # TODO
                # grad[j] = grad[j] + (y_i - temp) x_{ij}
                grad[j]=grad[j] + (y_tr[i]-temp)*(X_tr[i][j])


        for j in range(n_vars):
        	grad[j]=grad[j]-(lambda_value*w[j])
        # TODO
        # w_new = w + lr * grad
       
        w_new = w + (lr*grad)
       


        if iter%50 == 0:
            print('Iteration: {0}, mean abs gradient = {1}'.format(iter, np.mean(np.abs(grad))))

        # stopping criteria and perform update if not stopping
        if (np.mean(np.abs(grad)) < tolerance):
            w = w_new
            break
        else :
            w = w_new

        if (iter >= max_iter):
            break

    #Finding the test accuracy and train accuracy and returning the values
    test_accuracy  = accuracy(X_ts, y_ts, w)
    train_accuracy = accuracy(X_tr, y_tr, w)
    return test_accuracy, train_accuracy
def plot_line_chart(iterations,train_acc,test_acc):
	#Training and Testing Accuracy
	   plt.figure()
	   plt.xlabel('No of Iterations')
	   plt.ylabel('Accuracy')
	   plt.plot(iterations,train_acc, label = 'Training Accuracy')
	   plt.plot(iterations,test_acc, label = 'Testing Accuracy')
	   #plt.ylim([80,100])
	   plt.legend()
	   plt.show()
	   plt.close()
def plot_line_chart_regularized(k_values,train_acc,test_acc):
	#Training and Testing Accuracy
	   plt.figure()
	   plt.xlabel('K Values')
	   plt.ylabel('Accuracy')
	   plt.plot(k_values,train_acc, label = 'Training Accuracy')
	   plt.plot(k_values,test_acc, label = 'Testing Accuracy')
	   #plt.ylim([80,100])
	   plt.legend()
	   plt.show()
	   plt.close()
	   
# read files
D_tr = genfromtxt('spambasetrain.csv', delimiter = ',', encoding = 'utf-8')
D_ts = genfromtxt('spambasetest.csv', delimiter = ',', encoding = 'utf-8')

# construct x and y for training and testing
X_tr = D_tr[: ,: -1]
y_tr = D_tr[: , -1]
X_ts = D_ts[: ,: -1]
y_ts = D_ts[: , -1]

# number of training / testing samples
n_tr = D_tr.shape[0]
n_ts = D_ts.shape[0]

# add 1 as feature
X_tr = np.concatenate((np.ones((n_tr, 1)), X_tr), axis = 1)
X_ts = np.concatenate((np.ones((n_ts, 1)), X_ts), axis = 1)

# set learning rate
#lr = 1e-3
train_accuracy,test_accuracy=[],[]
#Defining learning rates 
learning_rate=[1e-0,1e-2,1e-4,1e-6]
#Defining no of iterations
Total_Iterations=[200,400,600,800,1000]
#Iterate through the values in learning rate's and finding train,test values
for i,lr in enumerate(learning_rate):
    train_test_values=logistic_reg(X_tr,X_ts,lr,Total_Iterations)
    train_accuracy.append(train_test_values[1])
    test_accuracy.append(train_test_values[0])
#Printing the accuracies for different values of learning rate's
for i in range(4):
    print('train accuracy = {0}, test_accuracy = {1} for Learning Rate = {2}'.format(str(train_accuracy[i][4]), str(test_accuracy[i][4]), str(learning_rate[i])))

#Plot the accuracies with respect to iterations
for i in range(4):
    plot_line_chart(Total_Iterations,train_accuracy[i],test_accuracy[i])
#Defining all the necessary values to compute regularized values
train_accuracy_regularized,test_accuracy_regularized=[],[]
lambda_values,k_values,lr=[],[],1e-3
#Appending K values for different iterations
for k in range(-8,4,2):
    lambda_values.append(2**k)
    k_values.append(k)
#finding accuracies for different lambda values
for i,lambda_value in enumerate(lambda_values):
    train_test_values_regularized=logistic_reg_regularized(X_tr,X_ts,lr,lambda_value)
    train_accuracy_regularized.append(train_test_values_regularized[1])
    test_accuracy_regularized.append(train_test_values_regularized[0])
#Printing the accuracies for different lambda values
for i in range(6):
    print('train accuracy = {0}, test_accuracy = {1} for Lambda value = {2}'.format(str(train_accuracy_regularized[i]), str(test_accuracy_regularized[i]), str(lambda_values[i])))
plot_line_chart_regularized(k_values,train_accuracy_regularized,test_accuracy_regularized)