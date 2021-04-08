<h1>Deep Learning- classification of gender by voice recognation</h1>

<h2>Abstract</h2>

The need to find a solution to the voice recognition problem, made it a popular topic
among the deep learning community. many companies started using voice
recognition related technologies like the popular google home, amazon Alexa and
Apple Siri. voice recognition has many sub-problems, one of them being gender
recognition. The costumer gender is a valuable information most companies will like
to attain in order to provide better customer service and better information for
Advertisers. This article will suggest one solution to the problem using a deep
learning approach; Long Short Term Memory (LSTM) and Recursive Neural Network
(RNN).

<h2>Introduction</h2>

Long Short-Term Memory (LSTM) is widely used in speech recognition. In order to
achieve higher prediction accuracy, machine learning scientists have built
increasingly larger models. Such large model is both computation intensive and
memory intensive. This project Show the use of RNN model with LSTM to classify
between male and female voices giving statistic data on the acoustic properties of
the voice and speech. The dataset was attained from Kaggle and was tested on
machine learning models (Random Forest, SVM, Logistic Regression and more) but
not deep learning models. The data contain 3168 voice recording with labels of male
or female and was pre-processed in to our feature using acoustic analysis in R and
using the seewave and tuneR packages and with analyzed frequency range of 0hz280hz (which is the human vocal range). in the article we also show The result of
logistic regression and multilayer perceptron models on the same dataset and we
show That the RNN gave more accurate result and manage to get there much faster
than the previous models
