#  Classify the MNIST digits using a one nearest neighbour classifier and Euclidean distance
import random
import numpy as np
import matplotlib.pyplot as plt
import mnist
class mnist_data:
    def sqDistance(self,p, q, pSOS, qSOS):
        #  Efficiently compute squared euclidean distances between sets of vectors

        #  Compute the squared Euclidean distances between every d-dimensional point
        #  in p to every d-dimensional point in q. Both p and q are
        #  npoints-by-ndimensions. 
        #  d(i, j) = sum((p(i, :) - q(j, :)).^2)

        d = np.add(pSOS, qSOS.T) - 2*np.dot(p, q.T)

        return d

    def main_function(self,trainvalue):

        np.random.seed(1)

        #  Set training & testing 
        Xtrain, ytrain, Xtest, ytest = mnist.load_data()

        train_size = trainvalue
        #test_size  = 10000

        Xtrain = Xtrain[0:train_size]
        ytrain = ytrain[0:train_size]

        #Xtest = Xtest[0:test_size]
        #ytest = ytest[0:test_size]

        #  Precompute sum of squares term for speed
        XtrainSOS = np.sum(Xtrain**2, axis=1, keepdims=True)
        XtestSOS  = np.sum(Xtest**2, axis=1, keepdims=True)

        #  fully solution takes too much memory so we will classify in batches
        #  nbatches must be an even divisor of test_size, increase if you run out of memory 
        if len(Xtest) > 1000:
          nbatches = 50
        else:
          nbatches = 5

        batches = np.array_split(np.arange(len(Xtest)), nbatches)
        ypred = np.zeros_like(ytest)

        #  Classify
        for i in range(nbatches):
            dst = self.sqDistance(Xtest[batches[i]], Xtrain, XtestSOS[batches[i]], XtrainSOS)
            closest = np.argmin(dst, axis=1)
            ypred[batches[i]] = ytrain[closest]

        #  Report
        errorRate = (ypred != ytest).mean()
        print('Error Rate: {:.2f}%\n'.format(100*errorRate))
        return 100*errorRate
        #  image plot
        #plt.imshow(Xtrain[0].reshape(28, 28), cmap='gray')
        #plt.show()

    def cross_validation_error(self,nbatches):
        np.random.seed(1)

        #  Set training & testing 
        Xtrain, ytrain, Xtest, ytest = mnist.load_data()

        #train_size = trainvalue
        #test_size  = 10000

        #Taking only 1st training 1000 Training Examples
        Xtrain = Xtrain[0:1000]
        ytrain = ytrain[0:1000]

        #Xtest = Xtest[0:test_size]
        #ytest = ytest[0:test_size]

        #Split the data into batches
        data_for_folds = np.array_split(np.arange(len(Xtrain)), nbatches)
        ypred = np.zeros_like(ytrain)

        #  Classify
        for i in range(nbatches):
            #Remove the test set
            X_train=np.delete(Xtrain,data_for_folds[i],0)
            #Test set
            X_test=Xtrain[data_for_folds[i]]
            #Labels
            y_train=np.delete(ytrain,data_for_folds[i],0)
            #Precompute sum of squares term for speed
            XtrainSOS = np.sum(X_train**2, axis=1, keepdims=True)
            XtestSOS  = np.sum(X_test**2, axis=1, keepdims=True)
            #Calculate the distance for testset
            dst = self.sqDistance(X_test, X_train, XtestSOS, XtrainSOS)
            #Arrange them according to minimum distance
            closest = np.argmin(dst, axis=1)
            ypred[data_for_folds[i]] = y_train[closest]

        #  Report
        errorRate = (ypred != ytrain).mean()
        print('Error Rate: {:.2f}%\n'.format(100*errorRate))
        return 100*errorRate
            

# Q1:  Plot a figure where the x-asix is number of training
#      examples (e.g. 100, 1000, 2500, 5000, 7500, 10000), and the y-axis is test error.

# TODO
    def plot_figure1(self,training_examples,test_error):
        #Plot the figure
            plt.figure()
        #Plot the xlabel
            plt.xlabel('Train Examples')
        #Plot the ylabel
            plt.ylabel('Test Error')
        #Plot the figure with training examples and test error
            plt.plot(training_examples,test_error,label ='TrainExamples vs TestError')
        #Set the ylimit
            plt.ylim([0,50])
            plt.legend()
        #Plot the curve
            plt.show()
        #close the plot
            plt.close()

# Q2:  plot the n-fold cross validation error for the first 1000 training examples

# TODO
    def plot_figure2(self,no_of_folds,cross_validation_error_for_different_n):
        #Plot the figure
            plt.figure()
        #Plot the xlabel 
            plt.xlabel('no of folds')
        #Plot the ylabel
            plt.ylabel('cross validation error')
        #Plot the figure
            plt.plot(no_of_folds,cross_validation_error_for_different_n,label ='no of folds vs Cross Validation Error')
        #Set the limit on Y-axis
            plt.ylim([0,20])
            plt.legend()
        #Plot the Figure
            plt.show()
        #Close the Figure
            plt.close()
#Create the Object for the class
mn=mnist_data()
#Initialize the Values
test_error=[]
training_examples=[100,1000,2500,5000,7500,10000]
#Iterate through Training Examples
for i in training_examples:
#Print the Test Error    
    print("test error")
#Appending the Test Error
    test_error.append(mn.main_function(i))
#Test Error List
print("Test Error")
print(test_error)
#Training Examples
print("Training Examples")
print(training_examples)
#Plot the Figure 
mn.plot_figure1(training_examples,test_error)

cross_validation_error_for_different_n=[]
#No of Folds
no_of_folds=[3,10,50,100,1000]
#Iterate through Folds
for i in no_of_folds:
#Append cross validation error for different n
   cross_validation_error_for_different_n.append(mn.cross_validation_error(i))
print("no of folds")
print(no_of_folds)
print("cross validation error for different n")
print(cross_validation_error_for_different_n)
#Plot the figure
mn.plot_figure2(no_of_folds,cross_validation_error_for_different_n)

