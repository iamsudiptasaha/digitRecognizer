# digitRecognizer
In this competition, your goal is to correctly identify digits from a dataset of tens of thousands of handwritten images. We’ve curated a set of tutorial-style kernels which cover everything from regression to neural networks. We encourage you to experiment with different algorithms to learn first-hand what works well and how techniques compare. 
Competition Link : https://www.kaggle.com/c/digit-recognizer
Get the latest dataset from the above mentioned link.


Kaggle Kernal Link : https://www.kaggle.com/iamsudiptasaha/knn-with-additional-data-preprocessing


<p>
Okay for before I begin, you might want you question why did I opt for KNN? 
Well, the top scores in leaderboard have been acquired mostly by using CNN (you can find numerous kernals!) 
and since I am still a novice in ML, I wanted to try out this model. Achieved a score of 0.973.
I have tried with ANN model, but KNN gave better accuracy!
</p>

<p>Four Step model -> Gather, Preprocess, Train, Predict</p>



========================-Gather========================

test_dataset  = pd.read_csv('../input/test.csv')  #test dataset
train_dataset = pd.read_csv('../input/train.csv') #train dataset



========================Preprocess========================



1. Create new set of features and produce threshold image.
2. Add to provided set of features and normalize the data by 255 (since binary image)
3. Use PCA (Principal Component Analysis) to reduce the number of features to a handful of highly essential ones.
4. Split data into training and test set.



========================Train========================



Use K-Nearest Neighbour Classifier 
knn = KNeighborsClassifier(n_neighbors=10, n_jobs=-1) #Here, n_jobs =-1 for the machine to use all cores!
knn.fit(X_train, y_train)   #fit the data
y_pred=knn.predict(X_test)  #Save the prediction



========================Predict========================



print("Accuracy of prediction : "+str(np.sum(exclusive)/len(exclusive)))

#Classification Report
print("Classification report for classifier %s:\n%s\n"
      % (knn, metrics.classification_report(expected, predicted)))
#Confusion Matrix!
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))




<b>Thank you !</b>
<p>Kaggle Kernal Link : https://www.kaggle.com/iamsudiptasaha/knn-with-additional-data-preprocessing</p>
<p>Special thanks to : https://www.kaggle.com/statinstilettos/neural-network-approach</p>
