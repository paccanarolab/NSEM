# NSEM
Non-negative self-expressive models

The code for training and testing our models is written in Python 3.0, Ubuntu 16.04 was used to execute these scripts (it will also run in Windows and OSx). The following dependencies (python modules and libraries) are needed to run our code:

●	argparse

●	scipy

●	numpy

●	collections

●	itertools

●	warnings

Brief description of the files in the repository:

●	runNSEM.py Python script to fully run NSEM. This includes training and testing of the model. It receives as input the following files and parameters:
    
    ○ Path to the files containing the training and testing matrices in the same format specified by SLIM. The parameters for the command line execution are -t (path to train) and -s (path to test).

    ○ L1 and L2 regularization parameters. The parameters for the command line execution are -b (L2- regularization parameter Beta) and -l (L1- regularization parameter Lambda).

    ○ Top N number to evaluate the performance of the model. The parameter for the command line execution is -n (topN).

●	Nsem.py Python Class that contains the necessary variables and methods to train and evaluate the models. An object of this class is instantiated in runNSEM.py. The parameters the class receives as input are:

    ○ Ytrain, the training matrix in sparse format.

    ○ Ytest, the testing matrix in sparse format.

    ○ tolX, the maximum tolerable change in W (convergence).

    ○ var, the variance of the initial random distribution.

    ○ max_iters, maximum number of iterations.

    ○ The L2- and L1- regularization parameters are passed to the method Nsem.train(L2beta, L1lambda).
The script generates as output: the Hit Rate and the Average Reciprocal Hit Rate, at the desired topN, as well as the trained matrix W.

●	We also provide a sample training matrix (train.mat) and testing matrix (test.mat) retrieved from the SLIM downloads website. They were generated from the ML100K dataset so the results should be similar to the ones described in the paper. 

A sample run of the script would be:

```
$ python runNSEM.py -t train.mat -s test.mat -b 0.5 -l 4 -n 10
```

We suggest using beta = 0.5 and lambda = 4 given that we’ve obtained the best results with those values for the regularization parameters.
A typical output should look like this:

Reading and loading training and testing matrices...
Done...
Training model...
Done...
Calculating HR and ARHR...
Statistics for model trained with L2Beta: 0.5, L2Lambda: 4.
Hit Rate at top_10: 0.3488865323435843
Average Reciprocal Hit Rate at top_10: 0.16067136292480924

Dumping results…
