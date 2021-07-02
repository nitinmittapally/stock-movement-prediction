# Summary of the project: Stock Movement Prediction

In this project I try to explore and understand the basics of building an LSTM netork in tensor flow. 
Stock data from 2015-2017 is used to train the network, 2018 is used for validation and 2019-2020 is used as test data. 
Used drop layers for parameter regularization. 

The daily close values of Intel, Amazon,and Facebook are used to get the daily returns which are then scaled to zero mean and unit variance.  
Simiarly, volumes of the above stocks are also converted in to z form. 

LSTM input layer of 7x6 is used and the output is feed to NN. 

An Overall acuracy of 53% was acheived.

