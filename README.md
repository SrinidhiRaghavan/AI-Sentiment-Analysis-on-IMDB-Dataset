# AI-Sentiment-Analysis-on-IMDB-Dataset


## Introduction ##

Given the availability of a large volume of online review data (Amazon, IMDB, etc.), sentiment analysis becomes increasingly important. In this project, a sentiment classifier is built which evaluates the polarity of a piece of text being either positive or negative. 

## Getting the Dataset ##

The "Large Movie Review Dataset"(*) shall be used for this project. The dataset is compiled from a collection of 50,000 reviews from IMDB on the condition there are no more than 30 reviews per movie. The numbers of positive and negative reviews are equal. Negative reviews have scores less or equal than 4 out of 10 while a positive review have score greater or equal than 7 out of 10. Neutral reviews are not included. The 50,000 reviews are divided evenly into the training and test set. 

The Training Dataset used is stored in the zipped folder: aclImbdb.tar file. This can also be downloaded from: http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz. 

The Test Dataset is stored in the folder named 'test'



## Data Preprocessing ##

The training dataset in aclImdb folder has two sub-directories pos/ for positive texts and neg/ for negative ones. Use only these two directories. The first task is to combine both of them to a single csv file, “imdb_tr.csv”. The csv file has three columns,"row_number" and “text” and “polarity”. The column “text” contains review texts from the aclImdb database and the column “polarity” consists of sentiment labels, 1 for positive and 0 for negative. The file imdb_tr.csv is an output of this preprocessing. In addition, common English stopwords should be removed. An English stopwords reference ('stopwords.en') is given in the code for reference.


## Data Representations Used  ##

Unigram , Bigram , TfIdf 


## Algorithmic Overview ##

In this project, we will train a Stochastic Gradient Descent Classifier. This is used instead of gradient descent as gradient descent is prohibitively expensive when the dataset is extremely large because every single data point needs to be processed.
SGD algorithm performs just as good with a small random subset of the original data. This is the central idea of Stochastic SGD and particularly handy for the text data since text corpus are often humongous.

A good description of this algorithm can be found at: https://en.wikipedia.org/wiki/Stochastic_gradient_descent.


## Functions used in the driver_3 file ##

imdb_data_preprocess : Explores the neg and pos folders from aclImdb/train and creates a imdb_tr.csv file in the required format

remove_stopwords : Takes a sentence and the stopwords as inputs and returns the sentence without any stopwords

unigram_process : Takes the data to be fit as the input and returns a vectorizer of the unigram as output

bigram_process : Takes the data to be fit as the input and returns a vectorizer of the bigram as output 

tfidf_process : Takes the data to be fit as the input and returns a vectorizer of the tfidf as output

retrieve_data : Takes a CSV file as the input and returns the corresponding arrays of labels and data as output

stochastic_descent : Applies Stochastic on the training data and returns the predicted labels

accuracy : Finds the accuracy in percentage given the training and test labels 

write_txt : Writes the given data to a text file 


## Environment ##

Language : Python 3

Libraries : Scikit, Pandas 


## How to Execute? ##

Run python driver_3.py


## Results ##

Output files are:

unigram.output

unigramtfidf.output

bigram.output

bigramtfidf.output

Here, 1 is given for positive labels and 0 is for negative labels 
