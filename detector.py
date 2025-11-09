import numpy as np # for numerical operations
import pandas as pd # for data manipulation
import itertools # for advanced iteration
from sklearn.model_selection import train_test_split # for splitting the dataset
from sklearn.feature_extraction.text import TfidfVectorizer # for text feature extraction
from sklearn.linear_model import PassiveAggressiveClassifier # for classification
from sklearn.metrics import accuracy_score, confusion_matrix # for model evaluation

df = pd.read_csv('news.csv') # read the dataset

print("Rows and Column Count :",df.shape) # check the shape of the dataset
print("---------------------------  ------------------  ------------------")
print("First 5 rows of the dataset :")
print(df.head()) # print the first 5 rows of the dataset
print("---------------------------  ------------------  ------------------")

labels = df.label # extract labels
print("First 5 labels :")
print(labels.head()) # print the first 5 labels
print("---------------------------  ------------------  ------------------")

x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7) # split the dataset into train and test sets

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7) # initialize the TF-IDF vectorizer

tfidf_train = tfidf_vectorizer.fit_transform(x_train) # fit and transform the training data
tfidf_test = tfidf_vectorizer.transform(x_test) # transform the test data

pac = PassiveAggressiveClassifier(max_iter=50) # initialize the Passive Aggressive Classifier
pac.fit(tfidf_train, y_train) # train the classifier

y_pred = pac.predict(tfidf_test) # make predictions on the test data
score = accuracy_score(y_test, y_pred) # calculate the accuracy
print(f'Accuracy: {round(score*100,2)}%') # print the accuracy
print("---------------------------  ------------------  ------------------")


confusion_mat = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']) # compute the confusion matrix
print('Confusion Matrix:\n', confusion_mat) # print the confusion matrix
print("---------------------------  ------------------  ------------------")
