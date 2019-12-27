import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.markers import MarkerStyle
from scipy.spatial.distance import cdist
from scipy import stats

#  KNN function
class irisNN:
    def knn_predict(self,X_test, X_train, y_train, k=3):
        n_X_test = X_test.shape[0]
        decision = np.zeros((n_X_test, 1))
        for i in range(n_X_test):
            point = X_test[[i],:]

            #  compute euclidan distance from the point to all training data
            dist = cdist(X_train, point)

            #  sort the distance, get the index
            idx_sorted = np.argsort(dist, axis=0)

            #  find the most frequent class among the k nearest neighbour
            pred = stats.mode(y_train[idx_sorted[0:k]])

            decision[i] = pred[0]
        

        return decision
    

    def knn_predict_train(self,random_value,X_train, y_train, k=3):
        n_X_test,count = X_train.shape[0],0
        decision = np.zeros((n_X_test, 1))
        for i in range(n_X_test):
            point = X_train[[i],:]
            X_train_remaining=np.delete(X_train,i,0)
            #print(len(X_train_remaining))
            y_train_remaining=np.delete(y_train,i,0)
            #  compute euclidan distance from the point to all training data
            dist = cdist(X_train_remaining, point)

            #  sort the distance, get the index
            idx_sorted = np.argsort(dist, axis=0)

            #  find the most frequent class among the k nearest neighbour
            pred = stats.mode( y_train_remaining[idx_sorted[0:k]])
            #Assign the highest related class to decision 1
            decision[i] = pred[0]
            #Check whether decision and train labels equal or not to find error rate
            if decision[i]!=y_train[i]:
            #Increment count
            	count=count+1
        #Calculate the errorrate
        errorrate=(count/150)
        print("No of changed labels is {0}".format(random_value))
        print("Training error rate is {0}".format(errorrate*100))
        

    def main_function(self,random_value):
        np.random.seed(1)
        #Initiate the array
        values_changed=[]
        #Append the values which are to be changed in 1st 150 examples
        for i in range(random_value):
            values_changed.append(random.randint(0,149))
        
        # Setup data
        D = np.genfromtxt('iris.csv', delimiter=',')
        X_train = D[:, 0:2]   # feature
        y_train = D[:, -1]    # label
        
        #Change the labels
        for j in values_changed:
            if y_train[j]==1:
                y_train[j]=2
            elif y_train[j]==2:
                y_train[j]=3
            else:
                y_train[j]=1

        #Calling the function to find training Error Rate
        self.knn_predict_train(random_value,X_train,y_train,3)
        # Setup meshgrid
        x1, x2 = np.meshgrid(np.arange(2,5,0.01), np.arange(0,3,0.01))
        X12 = np.c_[x1.ravel(), x2.ravel()]
        

        # Compute 1NN decision
        
        k = 3
        decision = self.knn_predict(X12, X_train, y_train, k)


        # Create color maps
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        # plot decisions in the grid
        
        decision = decision.reshape(x1.shape)
        plt.figure()
        plt.pcolormesh(x1, x2, decision, cmap=cmap_light)

        # Plot the training points
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, s=25)
        plt.xlim(x1.min(), x1.max())
        plt.ylim(x2.min(), x2.max())

        plt.show()
        plt.close()
#Create The object
iris=irisNN()
D = np.genfromtxt('iris.csv', delimiter=',')
X_train = D[:, 0:2]   # feature
y_train = D[:, -1]    # label
#iris.knn_predict_train(X_train,y_train,3)
for i in [10,20,30,50]:
    iris.main_function(i)