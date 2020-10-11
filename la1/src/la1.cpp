//============================================================================
// Introduction to computational models
// Name        : la1.cpp
// Author      : Pedro A. Gutiérrez
// Version     :
// Copyright   : Universidad de Córdoba
//============================================================================

#include <cstdlib> // To establish the seed srand() and generate pseudorandom numbers rand()
#include <ctime> // To obtain current time time()
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "mlp.hpp"
#include "util.h"

using namespace imc;
using namespace std;

int main(int argc, char** argv)
{
    // Process arguments of the command line
    bool Tflag = 0, wflag = 0, pflag = 0, tflag = 0, iflag = 0, hflag = 0, lflag = 0;
    char *trainFilename = NULL, *wvalue = NULL, *testFilename = NULL;
    int hiddenLayers, maxIterations, neuronsPerLayer;
    double validationRatio = .0, eta = 0.1, mu = 0.9, decrementFactor = 1.0;
    int c;

    opterr = 0;

    // a: Option that requires an argument
    // a:: The argument required is optional
    while ((c = getopt(argc, argv, "d:e:m:v:i:l:h:t:T:w:p")) != -1) {
        // The parameters needed for using the optional prediction mode of Kaggle have been included.
        // You should add the rest of parameters needed for the lab assignment.
        switch (c) {
        case 't':
            trainFilename = optarg;
            tflag = true;
            break;
        case 'l':
            hiddenLayers = atoi(optarg);
            lflag = true;
            break;
        case 'h':
            neuronsPerLayer = atoi(optarg);
            hflag = true;
            break;
        case 'i':
            maxIterations = atoi(optarg);
            iflag = true;
            break;
        case 'T':
            Tflag = true;
            testFilename = optarg;
            break;
        case 'w':
            wflag = true;
            wvalue = optarg;
            break;
        case 'p':
            pflag = true;
            break;
        case 'v':
            validationRatio = atof(optarg);
            break;
        case 'e':
            eta = atof(optarg);
            break;
        case 'm':
            mu = atof(optarg);
            break;
        case 'd':
            decrementFactor = atof(optarg);
            break;
        case '?':
            if (optopt == 'T' || optopt == 'w' || optopt == 'p' || optopt == 'i' || optopt == 'h' || optopt == 'l')
                fprintf(stderr, "The option -%c requires an argument.\n", optopt);
            else if (isprint(optopt))
                fprintf(stderr, "Unknown option `-%c'.\n", optopt);
            else
                fprintf(stderr, "Unknown character `\\x%x'.\n", optopt);
            return EXIT_FAILURE;
        default:
            return EXIT_FAILURE;
        }
    }

    if (!pflag) {
        //////////////////////////////////
        // TRAINING AND EVALUATION MODE //
        //////////////////////////////////

        // Multilayer perceptron object
        MLP mlp;

        // Parameters of the mlp. For example, mlp.eta = value;
        int iterations = iflag ? maxIterations : 1000; // This should be corrected

        // Read training and test data: call to mlp.readData(...)
        if (!tflag)
            perror("Specify the train dataset with th -t flag");

        Dataset* trainDataset = mlp.readData(trainFilename); // This should be corrected
        Dataset* testDataset = Tflag ? mlp.readData(testFilename) : trainDataset; // This should be corrected

        // Initialize topology vector
        int layers = lflag ? hiddenLayers : 1;
        int topology[] = { trainDataset->nOfInputs, hflag ? neuronsPerLayer : 5, trainDataset->nOfOutputs }; // This should be corrected

        // Initialize the network using the topology vector
        mlp.initialize(layers + 2, topology);

        mlp.eta = eta;
        mlp.mu = mu;
        mlp.decrementFactor = decrementFactor;
        mlp.validationRatio = validationRatio;
        // Seed for random numbers
        int seeds[] = { 1, 2, 3, 4, 5 };
        double* testErrors = new double[5];
        double* trainErrors = new double[5];
        double bestTestError = 1;
        for (int i = 0; i < 5; i++) {
            cout << "**********" << endl;
            cout << "SEED " << seeds[i] << endl;
            cout << "**********" << endl;
            srand(seeds[i]);
            mlp.runOnlineBackPropagation(trainDataset, testDataset, iterations, &(trainErrors[i]), &(testErrors[i]));
            cout << "We end!! => Final test error: " << testErrors[i] << endl;
            //mlp.predict(testDataset);

            // We save the weights every time we find a better model
            if (wflag && testErrors[i] <= bestTestError) {
                mlp.saveWeights(wvalue);
                bestTestError = testErrors[i];
            }
        }
        cout << "WE HAVE FINISHED WITH ALL THE SEEDS" << endl;

        double averageTestError = 0, stdTestError = 0;
        double averageTrainError = 0, stdTrainError = 0;

        util::getStatistics(testErrors, 5, averageTestError, stdTestError);
        util::getStatistics(trainErrors, 5, averageTrainError, stdTrainError);
        // Obtain training and test averages and standard deviations

        cout << "FINAL REPORT" << endl;
        cout << "************" << endl;
        cout << "Train error (Mean +- SD): " << averageTrainError << " +- " << stdTrainError << endl;
        cout << "Test error (Mean +- SD):          " << averageTestError << " +- " << stdTestError << endl;

        return EXIT_SUCCESS;
    } else {

        //////////////////////////////
        // PREDICTION MODE (KAGGLE) //
        //////////////////////////////

        // Multilayer perceptron object
        MLP mlp;

        // Initializing the network with the topology vector
        if (!wflag || !mlp.readWeights(wvalue)) {
            cerr << "Error while reading weights, we can not continue" << endl;
            exit(-1);
        }

        // Reading training and test data: call to mlp.readData(...)
        Dataset* testDataset;
        testDataset = mlp.readData(Tflag ? testFilename : trainFilename);
        if (testDataset == NULL) {
            cerr << "The test file is not valid, we can not continue" << endl;
            exit(-1);
        }

        mlp.predict(testDataset);

        return EXIT_SUCCESS;
    }
}