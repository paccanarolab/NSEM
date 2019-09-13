#/usr/bin/python3
"""
runNSEM.py: This is a script used to fully run the NSEM algorithm.
                 It receives as input the training and testing files, the regularization parameters,
                 and the topN to calculate the performance metrics.
Sample run: $ python runNSEM.py -t train.mat -s test.mat -b 0.5 -l 4 -n 10
More information about required dependencies and input and output files can be found in 
the SupplementaryMaterial.pdf file located in this folder.
"""


import argparse
from scipy import sparse
import NSEM

def read_csr_data(filename):    

    with open(filename,'r') as fi:
        reader = fi.readlines()
    data = list()
    # print 'Loading in memory...'
    for row,line in enumerate(reader):
        data.append((row,line.strip().split(' ')))
    
    users = list()
    items = list()
    values = list()
    for line in data:
        (row, cols) = line
        for i, elem in enumerate(cols):
            if i % 2 == 0:
                users.append(row)
                # substract 1 from the column index to respect python representation of matrices
                items.append(int(elem)-1)
                values.append(int(cols[i+1]))
            else:
                continue
  
    # print('Parsing done!')
    return users, items, values

def writeCsrToFile(sparseMatrix,fileName):
    with open(fileName,'w') as fi:
        for i, row in enumerate(sparseMatrix):
            line = list()
            for col in row.indices:
                line.append(str(col+1))
                line.append(str(sparseMatrix[i,col]))
            fi.write(' '.join(line)+'\n')

if __name__ == '__main__':
    aparser = argparse.ArgumentParser()
    required_arguments = aparser.add_argument_group('required arguments')
    required_arguments.add_argument(
        '-t','--training', 
        help='Path of the training matrix.',
        required=True)
    required_arguments.add_argument(
        '-s','--testing', 
        help='Path of the testing matrix.',
        required=True)
    required_arguments.add_argument(
        '-b','--beta', 
        help='Value of the L2 regularization parameter Beta.',
        required=True)
    required_arguments.add_argument(
        '-l','--lamb', 
        help='Value of the L1 regularization parameter Lambda.',
        required=True)
    required_arguments.add_argument(
        '-n','--topn', 
        help='Value of the topN to calculate Hit Rate (HR) and Average Reciprocal Hit Rate (ARHR).',
        choices=['5','10','15','20','25'],
        required=True)

    args = aparser.parse_args()

    print("Reading and loading training and testing matrices...")

    # We load the training matrix
    users, items, vals = read_csr_data(args.training)
    # Since indexes start at 0, we add 1 to obtain the number of users and items
    nUsers = max(users) + 1
    nItems = max(items) + 1
    Y = sparse.csr_matrix((vals, (users, items)), shape=(nUsers, nItems))
    # Start loading the testing matrix
    users, items, vals = read_csr_data(args.testing)
    YTest = sparse.csr_matrix((vals, (users, items)), shape=(nUsers, nItems))
    print("Done...")

    # Create an instance of the class Uslim with our default parameters
    mSLIM = NSEM.Mslim(Y, YTest, 1e-2, 0.01, 1000, 10000)

    # Extract the regularization parameters from command line
    l2Beta = float(args.beta)
    l1Lambda = float(args.lamb)
    topN = int(args.topn)

    # Train the model and save it in W
    print("Training model...")
    W = mSLIM.train(l2Beta,l1Lambda)
    print("Done...")

    # Calculate the Hit Rate (HR) and Average Reciprocal Hit Rate (ARHR)
    print("Calculating HR and ARHR...")
    hitRate = mSLIM.hit_rate(W,topN)
    avgRecHitRate = mSLIM.avg_rec_hit_rate(W, topN)

    # Printing statistics...
    print("Statistics for model trained with L2Beta: {}, L2Lambda: {}.".format(args.beta,args.lamb))
    print("Hit Rate at top_{}: {}".format(args.topn, str(hitRate)))
    print("Average Reciprocal Hit Rate at top_{}: {}".format(args.topn, str(avgRecHitRate)))

    # Dumping results in a text file
    print("\nDumping results...")
    with open("M-SLIM_BETA_{}_LAMBDA_{}_TOP_{}.txt".format(args.beta,args.lamb,args.topn),"w") as out:
        out.write("Statistics for model trained with L2Beta: {}, L2Lambda: {}.".format(args.beta,args.lamb))
        out.write("Hit Rate at top_{}: {}".format(args.topn, str(hitRate)))
        out.write("Average Reciprocal Hit Rate at top_{}: {}".format(args.topn, str(avgRecHitRate)))
    writeCsrToFile(W, "trained_W_beta_{}_lambda_{}.csr".format(args.beta,args.lamb))
