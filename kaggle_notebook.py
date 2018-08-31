# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



#Okay for before I begin, you might want you question why did I opt for KNN? 
#Well, the top scores in leaderboard are mostly using CNN (you can find numerous kernals!) 
#and since I am still a novice in ML, I wanted to try out this model. Achieved a score of 0.973.
#I have tried with ANN model, but KNN gave better accuracy!
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # for data visualization
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


#Four Step model -> Gather, Preprocess, Train, Predict
#lets first GATHER our datasets
test_dataset  = pd.read_csv('test.csv')  #test dataset
train_dataset = pd.read_csv('train.csv') #train dataset
y_test = pd.read_csv('sample_submission.csv')
test_image_id=y_test['ImageId']  #store the id for later use during score submission


#let's check the datasize
print("Train datasize : "+str(train_dataset.shape))
print("Test datasize : "+str(test_dataset.shape))


#You should get training dataset of shape (42000, 785)
#You should get test dataset of shape (28000, 784)
#Let's proceed if your the values matched!
#Next Step is to preprocess the data
#First let's visualize if the data is unbiased in accordance to the labels assigned
plt.hist(train_dataset['label'])
plt.title("Frequency Histogram of Labels in Training Data")
plt.xlabel("Label Code")
plt.ylabel("Frequency")
plt.show()

#Above data shows the data is more or less equally distributed among the levels, hence we are ready to proceed.
#Lets visualize the data provided, that is the pictorial representation of the data!

import math
from random import randint
# plot any random 25 digits from the training set. 
f, ax = plt.subplots(5, 5)
# plot some 4s as an example
for i in range(1,26):
    # Create a 1024x1024x3 array of 8 bit unsigned integers
    data = train_dataset.iloc[randint(0,42000),1:785].values #this is the first number
    nrows, ncols = 28, 28
    grid = data.reshape((nrows, ncols))
    n=math.ceil(i/5)-1
    m=[0,1,2,3,4]*5
    ax[m[i-1], n].imshow(grid)

#Run the above code multiple times to check out more pictures!
    
    
    
#Following is an additional preprocessing step that I included to enchance the prediction!
#What I have aimed to achieve is to get a more contrasting image of 2000 randomly selected digits 
#from each labelled data


X_dataset=train_dataset.copy()   #make a copy of the training dataset
train_size=X_dataset.shape[0]    #get the length of the actual training dataset
X_modtrain=pd.DataFrame()        #create a new dataframe to store new training data
X_modtrain=X_dataset[X_dataset['label']==0].sample(2000)   #randomly store 2000 datas labelled 0

#Following lines of code appends 2000 randomly sampled data from the original training set with different labels
X_modtrain=pd.concat((X_modtrain,X_dataset[X_dataset['label']==1].sample(2000))).reset_index(drop=True)
X_modtrain=pd.concat((X_modtrain,X_dataset[X_dataset['label']==2].sample(2000))).reset_index(drop=True)
X_modtrain=pd.concat((X_modtrain,X_dataset[X_dataset['label']==3].sample(2000))).reset_index(drop=True)
X_modtrain=pd.concat((X_modtrain,X_dataset[X_dataset['label']==4].sample(2000))).reset_index(drop=True)
X_modtrain=pd.concat((X_modtrain,X_dataset[X_dataset['label']==5].sample(2000))).reset_index(drop=True)
X_modtrain=pd.concat((X_modtrain,X_dataset[X_dataset['label']==6].sample(2000))).reset_index(drop=True)
X_modtrain=pd.concat((X_modtrain,X_dataset[X_dataset['label']==7].sample(2000))).reset_index(drop=True)
X_modtrain=pd.concat((X_modtrain,X_dataset[X_dataset['label']==8].sample(2000))).reset_index(drop=True)
X_modtrain=pd.concat((X_modtrain,X_dataset[X_dataset['label']==9].sample(2000))).reset_index(drop=True)

#Once we have gotten our required data, we want to preprocess the data, hence remove the label's column
X_modtrain_temp=X_modtrain.drop('label',1).copy()
#Set a threshold. I personally tested with 90 and 127. But 160 gave better accuracy.
threshold=160
#Modify the data, threshold the image to removes the outliers. 
#This however might negatively affect skewed images!
X_modtrain_temp[X_modtrain_temp<threshold]=0
X_modtrain_temp[X_modtrain_temp==threshold]=threshold
X_modtrain_temp[X_modtrain_temp>=threshold]=255

#Once additional training data is processed we add the labels column back to the modified data
X_modtrain_temp['label']=X_modtrain['label']
X_dataset=pd.concat((X_dataset, X_modtrain_temp)).reset_index(drop=True)


#Lets get the training and test set together
train_size=train_size+20000   #new training size

y_dataset=X_dataset['label']     #get the label's column
X_dataset=X_dataset.drop('label',1)   #remove the label's column

#Concatenating the train and test data
X_dataset=pd.concat((X_dataset, test_dataset)).reset_index(drop=True)     #concat the test data for further preprocessing

#Normalize the data. I tried thresholding, but normalization gave better accuracy with KNN.
X_dataset=X_dataset/255



#Further preprocessing
#Lets check if all the features are required or not.
#For this let's use PCA
#There are some good kernals available on LDA, and the t-SNE.

from sklearn import decomposition
from sklearn import datasets

#Let's first plot the data with different values of feature acceptance.

n_components=[0,25,50,75,100,150,200]
for i in n_components:
    pca = decomposition.PCA(n_components=i) #Finds first 200 PCs
    pca.fit(X_dataset[:(train_size)])
    plt.plot(pca.explained_variance_ratio_)
    
    
plt.ylabel('% of variance')

#In the above graph, we notice that the curve almost straightens at X=75. 
#So, 75 features are sufficiently adequate to build our model.

pca = decomposition.PCA(n_components=75) 
pca.fit(X_dataset[:(train_size)])

#Alter dataset accordingly
X_dataset = pd.DataFrame(pca.transform(X_dataset))

#You can always check the dataset before proceeding!
X_dataset.head()

#Spliting the independent values in test and training set.
PCtrain=X_dataset[:(train_size)]
PCtest =X_dataset[(train_size):]

#Uncomment the following code to develop your model on randomized data!

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(PCtrain, y_dataset, test_size=0.30, random_state=42)




#Import the model libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn import  metrics


knn = KNeighborsClassifier(n_neighbors=10, n_jobs=-1) #Here, n_jobs =-1 for the machine to use all cores!
knn.fit(X_train, y_train)   #fit the data
y_pred=knn.predict(X_test)  #Save the prediction

#Calculating Accuracy on the developed model!
expected = y_test.reset_index(drop=True)   #The expected output
predicted=pd.DataFrame(y_pred).reset_index(drop=True)  #The prediction acquired
exclusive=(predicted[0]==expected)  #Series to store if predictions have been successfull or not!
print("Accuracy : "+str(np.sum(exclusive)/len(exclusive)))

#Classification Report
print("Classification report for classifier %s:\n%s\n"
      % (knn, metrics.classification_report(expected, predicted)))
#Confusion Matrix!
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

#Working on the test data provided

X_train=PCtrain
y_train=y_dataset
X_test=PCtest


knn = KNeighborsClassifier(n_neighbors=10, n_jobs=-1) #Here, n_jobs =-1 for the machine to use all cores!
knn.fit(X_train, y_train)   #fit the new data
y_pred=knn.predict(X_test)  #new prediction

#Get the results 
predicted=pd.DataFrame(y_pred).reset_index(drop=True)  #The prediction acquired

#Save data!
output = pd.DataFrame()
output['ImageId']=test_image_id
output['Label']=predicted
output.to_csv('submission.csv',index=False)

#Voila! and we are done!

#I did like to thank this following kernal for helping me put up the code!
#https://www.kaggle.com/statinstilettos/neural-network-approach
