# Machine Learning - Natural Language Processing Assignments

This repo contains several exercises for the NLP course, which  concentrates on the study of deep learning techniques and their use in natural language processing. 

## First Assignment - Logistic Regression

The goal of this project is to implement gradient descent, stochastic gradient descent and mini-batch gradient descent methods for ridge regression, while also developing a sentiment classifier using logistic regression for the Twitter sentiment classification dataset.

For the gradient descent, we re going to implement the algorithms by defining the necessary functions, and then demonstrate their performance
using a specific dataset, and appropriate visualization techniques. All of the code is written in the notebook [GradientDescent](hw1/GradientDescent.ipynb), along with various
plots and comments.

For the classifier, We are going to combine a plethora of features, and see how it performs on the given dataset, aiming to high accuracy. We are going to use tools from Scikit-Learn. Afterwards, we are going to evaluate the classifier using several metrics. All of the code is written in the notebook [sentiment_classification](hw1/sentiment_classification.ipynb), along with various plots, comments, and metrics to evaluate the classifier.

## Second Assignment - Sentiment Classifier using Feed-Forward Neural Networks

The goal of this project is to develop a sentiment classifier using feed-forward neural networks for the Twitter sentiment analysis dataset. 

Two different python notebooks were created, and can be found in the parent directory. I chose to construct models using 2 different types of features:

 - TF-IDF features
 - GloVe features, using the pre-trained vectors
  
Both of the notebooks are well-documented. In general, the model which used GloVe (along with an embedding layer), behaved better than the TF-IDF one. All the models are compared using F1, accuracy, precision and recall, and structured the appropriate plots.

The notebook for this project can be found in the [sentiment_NN_Glove](hw2/Sentiment_NNs_GloVe.ipynb) and the [sentiment_NN_TFIDF](hw2/Sentiment_NNs_TFIDF.ipynb) notebooks.

## Third Assignment - Sentiment Classifier using stacked RNN with LSTM/GRU cells

The goal of this project is to develop a sentiment classifier using a bidirectional stacked RNN with LSTM/GRU cells for the Twitter sentiment analysis dataset. For the development of the models, there were experiments with the number of stacked RNNs, the number of hidden layers, type of cells, skip connections, gradient clipping and dropout probability. Also, an attention layer has been added, as well as a pooling one, and the best implemented model has been tested with those layers. The results, as well as the code can be found in the [Sentiment_RNNs](hw3/Sentiment_RNN.ipynb) notebook.

## Fourth Assignment - Document Retrieval and Textual Question Answering

There are 2 side projects in this project. The first one, aims in developing a document retrieval system to return titles of scientific
papers containing the answer to a given user question. The second one, aims in building a BERT-based model which returns “an answer”, given a user question and a passage which includes the answer of the question, given a pre-trained BERT model.

The notebook of the first one, can be found [here](hw4/Information_Retireval.ipynb), and for the second one [here](/hw4/Textual_QA.ipynb)