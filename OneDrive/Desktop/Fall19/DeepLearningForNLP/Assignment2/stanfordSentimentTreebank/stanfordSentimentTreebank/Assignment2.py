# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 22:34:32 2019

@author: srika
"""
'''import string

datasetlist=[]
with open("datasetSentences.txt", "rt", encoding = "latin-1") as datafile:
    for line in datafile:
        l = line.strip("")
        l = l.replace(" ","")
        l=l.lower()
        l = l.translate(str.maketrans('', '', string.punctuation))
        datasetlist.append(l)
#print(datasetlist)

originalsnippetslist=[]
with open("original_rt_Snippets.txt", "rt", encoding = "latin-1") as snippetfile:
    for line in snippetfile:
        l = line.strip("")
        l = l.replace(" ","")
        l=l.lower()
        l = l.translate(str.maketrans('', '', string.punctuation))
        originalsnippetslist.append(l)
#print(originalsnippetslist)


positivelist=[]
with open("rt-polarity_pos.pos", "rt", encoding = "latin-1") as positivefile:
    for line in positivefile:
        l = line.strip("")
        l = l.replace(" ","")
        l=l.lower()
        l = l.translate(str.maketrans('', '', string.punctuation))
        positivelist.append(l)
#print(positivelist)
        
negativelist=[]
with open("rt-polarity_neg.neg", "rt", encoding = "latin-1") as negativefile:
    for line in negativefile:
        l = line.strip("")
        l = l.replace(" ","")
        l=l.lower()
        l = l.translate(str.maketrans('', '', string.punctuation))
        negativelist.append(l)
print(negativelist)'''

import pandas as pd
import numpy as np
import re
snippet_file  = 'original_rt_snippets.txt'
positive_file = 'rt-polarity_pos.pos'
negative_file = 'rt-polarity_neg.neg'
sentences_for_data = pd.read_csv('datasetSentences.txt',sep = "\t",header = 0)
splitdata = pd.read_csv('datasetSplit.txt', delimiter = ",", header = 0)

snippet_file_list = []
filesnippet = open("original_snipp.txt","w+")
filesnippet.seek(0)
filesnippet.truncate()
with open(snippet_file, 'rt', encoding = 'latin-1') as f:
    for each_line in f:
        each_line = each_line.lower()                     
        each_line = re.sub('<.*?>', '',each_line) 
        each_line = re.sub('<[a-zA-Z]','',each_line)      
        each_line = re.sub(r'[^\w]','',each_line)         
        each_line = re.sub(r'[0-9]','',each_line)
        each_line = re.sub(r'[^\x00-\x7f]','',each_line)  
        snippet_file_list.append(each_line)
        filesnippet.write(each_line+"\n")
filesnippet.close()

       
positivefile=[]
fileforpositive = open("Positive_Labels.txt","w+")
fileforpositive.seek(0)
fileforpositive.truncate()
with open(positive_file,'rt',encoding='latin-1') as f:
    for each_line in f:
        each_line = each_line.lower()                     
        each_line = re.sub('<.*?>', '',each_line)         
        each_line = re.sub('<[a-zA-Z]','',each_line)      
        each_line = re.sub(r'[^\w]','',each_line)
        each_line = re.sub(r'[0-9]','',each_line)         
        each_line = re.sub(r'[^\x00-\x7f]','',each_line)  # removes non ascii characters
        positivefile.append(each_line)
        fileforpositive.write(each_line+"\n")
fileforpositive.close()

negativefile=[]
filefornegative = open("negative_labels.txt","w+")
filefornegative.seek(0)
filefornegative.truncate()
with open(negative_file,'rt',encoding='latin-1') as f:
    for each_line in f:
        each_line = each_line.lower()
        each_line = re.sub('<.*?>', '',each_line)
        each_line = re.sub('<[a-zA-Z]','',each_line)        
        each_line = re.sub(r'[^\w]','',each_line)
        each_line = re.sub(r'[0-9]','',each_line)
        each_line = re.sub(r'[^\x00-\x7f]','',each_line)
        negativefile.append(each_line)
        filefornegative.write(each_line+"\n")
filefornegative.close()

#Matching the labels for testing from positive file and negative file
labels_for_test = []
for line in snippet_file_list:
    if any(line == text for text in positivefile):
        labels_for_test.append(1)
    elif any(line == text for text in negativefile):
        labels_for_test.append(0)
    #Through by lot of preprocessing process we get to know that there is one sentence which doesn't belong to any of the files 
    elif line =="ihavenoproblemwithdifficultmoviesormoviesthatasktheaudiencetomeetthemhalfwayandconnectthedotsinsteadofhavingthingsallspelledoutbutfirstyouhavetogivetheaudienceareasontowanttoputforthateffort" :
        labels_for_test.append(0)
    else:
        print (line )

#To check with originalsnippitfile where it got matched
def IndexPositionWithOriginalFile(each_sentence, original_snippet_sentences):
    for indexposition in range(len(original_snippet_sentences)):
        if each_sentence in original_snippet_sentences[indexposition]:
            return indexposition      

#The below given sentence is the one which is not in either of those
# i have no problem with " difficult " movies , or movies that ask the audience to meet them halfway and connect the dots instead of having things all spelled out . but first , you have to give the audience a reason to <b>want</b> to put for that effort , and " i
# Hence adding the label manually.

#  Process the dataSentences that can be used to compare with original_rt_snippets
def filtering_sentence(each_sentence):
    each_sentence= each_sentence.lower()
    each_sentence = re.sub(r'[^\w]','',each_sentence)
    each_sentence = re.sub(r'[0-9]','',each_sentence)
    each_sentence = re.sub(r'[^\x00-\x7f]','',each_sentence)
    each_sentence = each_sentence.replace("lrb", "")
    each_sentence = each_sentence.replace("rrb","")
    return each_sentence



# Returns the labels for each sentence in dataSentences
labels=[]
filefordataset = open("dataset.txt","w+")
for each_sentence in list(sentences_for_data["sentence"].values):
    each_sentence= filtering_sentence(each_sentence)
    filefordataset.write(each_sentence+"\n")
    index = IndexPositionWithOriginalFile(each_sentence,snippet_file_list)
    labels.append(labels_for_test[index])
    snippet_file_list[index]= snippet_file_list[index].replace(each_sentence,"$$",1)  # Removes the sentence that has been matched already to avoid confusion and match with the correct original_snippet when same set of characters appear again.
filefordataset.close()
sentences_for_data["labels"]=labels

print(sentences_for_data)



