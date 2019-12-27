import re
import random
import math
import numpy as np
import matplotlib.pyplot as plt
#Creating class to execute functions inside it
class spam:
	def spammain(self):
		training_and_testing,i_values=[],[]
		self.spamcollection(0.1)
		for i in range(-5,1,1):
			training_and_testing.append(self.spamcollection(2**(i)))
			i_values.append(i)
		for i in range(len(training_and_testing)):
			train_acc,test_acc,fscore_train_acc,fscore_test_acc=[],[],[],[]
			for j in training_and_testing:
				train_acc.append(j[:1])
				test_acc.append(j[1])
				fscore_train_acc.append(j[2])
				fscore_test_acc.append(j[3])
			self.plot_line_chart(i_values,train_acc,test_acc,fscore_train_acc,fscore_test_acc)
#Plot function to draw training_accuracy/testing accuracy and fscore_train_accuracy/fscore_test_accuracy
	def plot_line_chart(self,i_values,train_acc,test_acc,fscore_train_acc,fscore_test_acc):
		#Training and Testing Accuracy
	    plt.figure()
	    plt.xlabel('Alpha Values')
	    plt.ylabel('Accuracy')
	    plt.plot(i_values,train_acc, label = 'Training Accuracy')
	    plt.plot(i_values, test_acc, label = 'Testing Accuracy')
	    plt.ylim([90,100])
	    plt.legend()
	   
	   	#F-Score Testing and Training
	    plt.figure()
	    plt.xlabel('Alpha Values')
	    plt.ylabel('F_Score')
	    plt.plot(i_values,fscore_train_acc, label = 'Training F_scores')
	    plt.plot(i_values, fscore_test_acc, label = 'Testing F_scores')
	    plt.ylim([0.6,1])
	    plt.legend()
	    plt.show()
	    plt.close()

	def spamcollection(self,alpha):
		random.seed(10)
		"""
		Read text data from file and pre-process text by doing the following
		1. convert to lowercase
		2. convert tabs to spaces
		3. remove "non-word" characters
		Store resulting "words" into an array
		"""
		FILENAME='SMSSpamCollection'
		all_data = open(FILENAME).readlines()

		# split into train and test
		num_samples = len(all_data)
		all_idx = list(range(num_samples))
		random.shuffle(all_idx)
		idx_limit = int(0.8*num_samples)
		train_idx = all_idx[:idx_limit]
		test_idx = all_idx[idx_limit:]
		train_examples = [all_data[ii] for ii in train_idx]
		test_examples = [all_data[ii] for ii in test_idx]


		# Preprocess train and test examples
		train_words = []
		train_labels = []
		test_words = []
		test_labels = []

		# train examples
		for line in train_examples:
		    line = line.strip('\r\n\t ')  # remove trailing spaces, tabs and carraige return
		    line = line.lower()  # lowercase
		    line = line.replace("\t", ' ')  # convert tabs to space
		    line_words = re.findall(r'\w+', line)
		    line_words = [xx for xx in line_words if xx != '']  # remove empty words

		    label = line_words[0]
		    label = 1 if label == 'spam' else 0
		    line_words = line_words[1:]
		    train_words.append(line_words)
		    train_labels.append(label)
		#print(train_words)
		#print(train_labels)  
		# test examples
		for line in test_examples:
		    line = line.strip('\r\n\t ')  # remove trailing spaces, tabs and carraige return
		    line = line.lower()  # lowercase
		    line = line.replace("\t", ' ')  # convert tabs to spae
		    line_words = re.findall(r'\w+', line)
		    line_words = [xx for xx in line_words if xx != '']  # remove empty words

		    label = line_words[0]
		    label = 1 if label == 'spam' else 0

		    line_words = line_words[1:]
		    test_words.append(line_words)
		    test_labels.append(label)

		spam_words = []
		ham_words = []
		alpha = alpha
		for ii in range(len(train_words)):  # we pass through words in each (train) SMS
		    words = train_words[ii]
		    label = train_labels[ii]
		    if label == 1:
		        spam_words += words
		    else:
		        ham_words += words
		input_words = spam_words + ham_words  # all words in the input vocabulary

		# Count spam and ham occurances for each word
		spam_counts = {}; ham_counts = {}
		# Spamcounts
		for word in spam_words:
		    try:
		        word_spam_count = spam_counts.get(word)
		        spam_counts[word] = word_spam_count + 1
		    except:
		        spam_counts[word] = 1 + alpha  # smoothening

		for word in ham_words:
		    try:
		        word_ham_count = ham_counts.get(word)
		        ham_counts[word] = word_ham_count + 1
		    except:
		        ham_counts[word] = 1 + alpha  # smoothening

		num_spam = len(spam_words)
		num_ham = len(ham_words)
		#print((spam_counts['hillsborough'] + alpha) / (num_spam + alpha * 20000))  # prob of "free" | spam
		#print((ham_counts['hillsborough'] + alpha) / (num_ham + alpha * 20000))  # prob of "free" | ham
		#Finding the prediction labels for test data
		prediction_labels=[]
		for i in test_words:
			p_spam,p_ham=float(1),float(1)
			for j in i:
				#print(spam_counts[j])
				if j in spam_counts:
					p_spam = p_spam*((spam_counts[j] + alpha)/(num_spam + alpha*20000))
				else:
					p_spam=p_spam*((0 + alpha)/(num_spam + alpha*20000))
				if j in ham_counts:
					p_ham = p_ham*((ham_counts[j] + alpha)/(num_ham + alpha*20000))
				else:
					p_ham= p_ham*((0 + alpha)/(num_ham + alpha*20000))
			if ((p_spam)*(num_spam))/(num_spam+num_ham)>=((p_ham)*(num_ham))/(num_spam+num_ham):
				prediction_labels=prediction_labels+[1]
			else:
				prediction_labels=prediction_labels+[0]
		#Testing accuracy is (no of correct predictions)/((no of correct predictions) + (no of incorrect predictions))
		no_of_correct_predictions,i=0,0
		while i<len(prediction_labels):
			if prediction_labels[i]==test_labels[i]:
				no_of_correct_predictions+=1
			i+=1
		Testing_Accuracy=((no_of_correct_predictions)/(len(prediction_labels)) * 100)
		if alpha==0.1:
			print("Testing Accuracy")
			print(Testing_Accuracy)
		prediction_labels_train=[]
		for i in train_words:
			p_spam_train,p_ham_train=float(1),float(1)
			for j in i:
				#print(spam_counts[j])
				if j in spam_counts:
					p_spam_train = p_spam_train*((spam_counts[j] + alpha)/(num_spam + alpha*20000))
				else:
					p_spam_train=p_spam_train*((0 + alpha)/(num_spam + alpha*20000))
				if j in ham_counts:
					p_ham_train = p_ham_train*((ham_counts[j] + alpha)/(num_ham + alpha*20000))
				else:
					p_ham_train= p_ham_train*((0 + alpha)/(num_ham + alpha*20000))
			if p_spam_train>=p_ham_train:
				prediction_labels_train=prediction_labels_train+[1]
			else:
				prediction_labels_train=prediction_labels_train+[0]
		#Testing accuracy is (no of correct predictions)/((no of correct predictions) + (no of incorrect predictions))
		no_of_correct_predictions,i=0,0
		while i<len(prediction_labels_train):
			if prediction_labels_train[i]==train_labels[i]:
				no_of_correct_predictions+=1
			i+=1
		#print("Training Accuracy")
		Training_Accuracy=((no_of_correct_predictions)/(len(prediction_labels_train)) * 100)
		#Training accuracy for train words
		True_Positive_train,False_Positive_train,False_Negative_train,True_Negative_train,i=0,0,0,0,0
		while i<len(prediction_labels_train):
			if prediction_labels_train[i]==train_labels[i] and train_labels[i]==1:
				True_Positive_train+=1
			if prediction_labels_train[i]==train_labels[i] and train_labels[i]==0:
				True_Negative_train+=1
			if prediction_labels_train[i]!=train_labels[i] and train_labels[i]==1:
				False_Negative_train+=1
			if prediction_labels_train[i]!=train_labels[i] and train_labels[i]==0:
				False_Positive_train+=1
			i+=1
		#print("Confusion Matrix train")
		#print("True_Positive_train","False_Positive_train","False_Negative_train","True_Negative_train")
		#print(True_Positive_train,False_Positive_train,False_Negative_train,True_Negative_train)

		#Precision = (True Positive)/((true positive) + (false positive))

		Precision_train=(True_Positive_train)/((True_Positive_train) + (False_Positive_train))
		#print("Precision train")
		#print(Precision_train)

		#Recall = (True Positive)/((True Positive) + (False Negative))

		Recall_train=(True_Positive_train)/((True_Positive_train) + (False_Negative_train))
		#print("Recall train")
		#print(Recall_train)

		#F-Score = 2(Precision)(Recall)/((Precision) + (Recall))

		FScore_train=(2)*(Precision_train)*(Recall_train)/((Precision_train)+(Recall_train))
		#print("F-Score train")
		#print(FScore_train)


		#Confusion Matrix
		True_Positive,False_Positive,False_Negative,True_Negative,i=0,0,0,0,0
		while i<len(prediction_labels):
			if prediction_labels[i]==test_labels[i] and test_labels[i]==1:
				True_Positive+=1
			if prediction_labels[i]==test_labels[i] and test_labels[i]==0:
				True_Negative+=1
			if prediction_labels[i]!=test_labels[i] and test_labels[i]==1:
				False_Negative+=1
			if prediction_labels[i]!=test_labels[i] and test_labels[i]==0:
				False_Positive+=1
			i+=1
		if alpha==0.1:	
			print("Confusion Matrix")
			print("True_Positive","False_Positive","False_Negative","True_Negative")
			print(True_Positive,False_Positive,False_Negative,True_Negative)

		#Precision = (True Positive)/((true positive) + (false positive))

		Precision=(True_Positive)/((True_Positive) + (False_Positive))
		if alpha==0.1:
			print("Precision")
			print(Precision)

		#Recall = (True Positive)/((True Positive) + (False Negative))

		Recall=(True_Positive)/((True_Positive) + (False_Negative))
		if alpha==0.1:
			print("Recall")
			print(Recall)

		#F-Score = 2(Precision)(Recall)/((Precision) + (Recall))

		FScore=((2)*(Precision)*(Recall))/((Precision)+(Recall))
		if alpha==0.1:
			print("F-Score")
			print(FScore)
		return [Training_Accuracy,Testing_Accuracy,FScore_train,FScore]

spm=spam()
spm.spammain()



