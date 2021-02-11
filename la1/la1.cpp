//============================================================================
// Introduction to computational models
// Name        : la1.cpp
// Author      : Pedro A. Gutiérrez
// Version     :
// Copyright   : Universidad de Córdoba
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <ctime>   // To obtain current time time()
#include <cstdlib> // To establish the seed srand() and generate pseudorandom numbers rand()
#include <string.h>
#include <math.h>

#include "MultilayerPerceptron.h"
#include "util.h"

using namespace imc;
using namespace std;
using namespace util;

int main(int argc, char **argv)
{
    // Process arguments of the command line
    bool pflag = 0, wflag = 0, Tflag = 0;
    char *testFilename = NULL, *trainFilename = NULL, *weightsFilename = NULL;
    int maxIter = 1000, neuronsPerLayer = 5, hiddenLayers = 1;
    double eta = .1, mu = .9, validationRatio = .0, decrementFactor = 1;
    int c;

    opterr = 0;

    // a: Option that requires an argument
    // a:: The argument required is optional
    while ((c = getopt(argc, argv, "t:T:i:h:l:e:m:v:d:w:p")) != -1)
    {
        // The parameters needed for using the optional prediction mode of Kaggle have been included.
        // You should add the rest of parameters needed for the lab assignment.
        switch (c)
        {
        case 't':
            trainFilename = optarg;
            break;
        case 'T':
            Tflag = true;
            testFilename = optarg;
            break;
        case 'i':
            maxIter = atoi(optarg);
            break;
        case 'h':
            neuronsPerLayer = atoi(optarg);
            break;
        case 'l':
            hiddenLayers = atoi(optarg);
            break;
        case 'e':
            eta = atof(optarg);
            break;
        case 'm':
            mu = atof(optarg);
            break;
        case 'v':
            validationRatio = atof(optarg);
            break;
        case 'd':
            decrementFactor = atof(optarg);
            break;
        case 'w':
            wflag = 1;
            weightsFilename = optarg;
            break;
        case 'p':
            pflag = true;
            break;
        case '?':
            if (optopt == 'T' || optopt == 'w' || optopt == 'p')
                fprintf(stderr, "The option -%c requires an argument.\n", optopt);
            else if (isprint(optopt))
                fprintf(stderr, "Unknown option `-%c'.\n", optopt);
            else
                fprintf(stderr,
                        "Unknown character `\\x%x'.\n",
                        optopt);
            return EXIT_FAILURE;
        default:
            return EXIT_FAILURE;
        }
    }

    if (!pflag)
    {
        //////////////////////////////////
        // TRAINING AND EVALUATION MODE //
        //////////////////////////////////

        // Multilayer perceptron object
        MultilayerPerceptron mlp;

        // Parameters of the mlp. For example, mlp.eta = value;
        int iterations = maxIter;

        mlp.eta = eta;
        mlp.decrementFactor = decrementFactor;
        mlp.mu = mu;
        mlp.validationRatio = validationRatio;

        // Read training and test data: call to mlp.readData(...)
        Dataset *trainDataset = mlp.readData(trainFilename);
        Dataset *testDataset = testFilename != NULL ? mlp.readData(testFilename) : trainDataset;

        // Initialize topology vector

        int *topology = new int[hiddenLayers + 2];
        topology[0] = trainDataset->nOfInputs;
        for (int i = 1; i < (hiddenLayers + 2 - 1); i++)
            topology[i] = neuronsPerLayer;
        topology[hiddenLayers + 2 - 1] = trainDataset->nOfOutputs;

        mlp.initialize(hiddenLayers + 2, topology);
        delete[] topology;

        // Seed for random numbers
        int seeds[] = {1, 2, 3, 4, 5};
        double *testErrors = new double[5];
        double *trainErrors = new double[5];
        double bestTestError = 1;
        for (int i = 0; i < 5; i++)
        {
            cout << "**********" << endl;
            cout << "SEED " << seeds[i] << endl;
            cout << "**********" << endl;
            srand(seeds[i]);
            mlp.runOnlineBackPropagation(trainDataset, testDataset, iterations, &(trainErrors[i]), &(testErrors[i]));
            cout << "We end!! => Final test error: " << testErrors[i] << endl;

            // We save the weights every time we find a better model
            if (wflag && testErrors[i] <= bestTestError)
            {
                mlp.saveWeights(weightsFilename);
                bestTestError = testErrors[i];
            }
        }

        cout << "WE HAVE FINISHED WITH ALL THE SEEDS" << endl;

        double averageTestError = 0, stdTestError = 0;
        double averageTrainError = 0, stdTrainError = 0;

        // Obtain training and test averages and standard deviations

        getStatistics(testErrors, 5, averageTestError, stdTestError);
        getStatistics(trainErrors, 5, averageTrainError, stdTrainError);

        delete[] trainErrors;
        delete[] testErrors;
        delete trainDataset;

        if (Tflag)
            delete testDataset;

        cout << "FINAL REPORT" << endl;
        cout << "************" << endl;
        cout << "Train error (Mean +- SD): " << averageTrainError << " +- " << stdTrainError << endl;
        cout << "Test error (Mean +- SD):          " << averageTestError << " +- " << stdTestError << endl;
        return EXIT_SUCCESS;
    }
    else
    {

        //////////////////////////////
        // PREDICTION MODE (KAGGLE) //
        //////////////////////////////

        // Multilayer perceptron object
        MultilayerPerceptron mlp;

        // Initializing the network with the topology vector
        if (!wflag || !mlp.readWeights(weightsFilename))
        {
            cerr << "Error while reading weights, we can not continue" << endl;
            exit(-1);
        }

        // Reading training and test data: call to mlp.readData(...)
        Dataset *testDataset;
        testDataset = mlp.readData(testFilename);
        if (testDataset == NULL)
        {
            cerr << "The test file is not valid, we can not continue" << endl;
            exit(-1);
        }

        mlp.predict(testDataset);

        delete testDataset;
        return EXIT_SUCCESS;
    }
}
