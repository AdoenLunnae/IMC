//============================================================================
// Introducción a los Modelos Computacionales
// Name        : practica1.cpp
// Author      : Pedro A. Gutiérrez
// Version     :
// Copyright   : Universidad de Córdoba
//============================================================================

#include <cstdlib> // To establish the seed srand() and generate pseudorandom numbers rand()
#include <ctime> // To obtain current time time()
#include <float.h> // For DBL_MAX
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "MultilayerPerceptron.h"
#include "util.h"

using namespace imc;
using namespace std;

int main(int argc, char** argv)
{
    // Process the command line
    bool Tflag = 0, tflag = 0, sflag = 0, wflag = 0, pflag = 0, oflag;
    char *trainFile = NULL, *testFile = NULL, *weightsFile = NULL;
    int c;
    opterr = 0;

    //Default values
    int hiddenLayers = 1, neuronsPerLayer = 5, maxIter = 1000, errorFunction = 0;
    double eta = .1, mu = .9, decrementFactor = 1.0, validationRatio = 0.0;

    // a: Option that requires an argument
    // a:: The argument required is optional
    while ((c = getopt(argc, argv, "osf:h:l:i:d:m:e:t:T:v:d:w:p")) != -1) {

        // The parameters needed for using the optional prediction mode of Kaggle have been included.
        // You should add the rest of parameters needed for the lab assignment.
        switch (c) {
        case 's':
            sflag = true;
            break;
        case 'o':
            oflag = true;
            break;
        case 'f':
            errorFunction = atoi(optarg);
            break;
        case 'h':
            neuronsPerLayer = atoi(optarg);
            break;
        case 'l':
            hiddenLayers = atoi(optarg);
            break;
        case 'i':
            maxIter = atoi(optarg);
            break;
        case 'd':
            decrementFactor = atof(optarg);
            break;
        case 'm':
            mu = atof(optarg);
            break;
        case 'e':
            eta = atof(optarg);
            break;
        case 't':
            tflag = true;
            trainFile = optarg;
            break;
        case 'T':
            Tflag = true;
            testFile = optarg;
            break;
        case 'w':
            wflag = true;
            weightsFile = optarg;
            break;
        case 'p':
            pflag = true;
            break;
        case 'v':
            validationRatio = atof(optarg);
            break;
        case '?':
            if (optopt == 'T' || optopt == 'w' || optopt == 'p')
                fprintf(stderr, "La opción -%c requiere un argumento.\n", optopt);
            else if (isprint(optopt))
                fprintf(stderr, "Opción desconocida `-%c'.\n", optopt);
            else
                fprintf(stderr,
                    "Caracter de opción desconocido `\\x%x'.\n",
                    optopt);
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
        MultilayerPerceptron mlp;

        // Parameters of the mlp. For example, mlp.eta = value

        // Read training and test data: call to mlp.readData(...)
        if (!tflag) {
            fprintf(stderr, "Please specify a training dataset with option '-t'");
            return EXIT_FAILURE;
        }

        Dataset* trainDataset = mlp.readData(trainFile);
        Dataset* testDataset = Tflag ? mlp.readData(testFile) : trainDataset;

        // Initialize topology vector
        int* topology = new int[hiddenLayers + 2];
        topology[0] = trainDataset->nOfInputs;
        for (int i = 1; i < (hiddenLayers + 2 - 1); i++)
            topology[i] = neuronsPerLayer;
        topology[hiddenLayers + 2 - 1] = trainDataset->nOfOutputs;
        mlp.initialize(hiddenLayers + 2, topology);

        delete[] topology;

        mlp.eta = eta;
        mlp.mu = mu;

        mlp.online = oflag;
        mlp.decrementFactor = decrementFactor;
        mlp.outputFunction = sflag ? 1 : 0;
        mlp.validationRatio = validationRatio;

        // Seed for random numbers
        int seeds[] = { 1, 2, 3, 4, 5 };
        double* trainErrors = new double[5];
        double* testErrors = new double[5];
        double* trainCCRs = new double[5];
        double* testCCRs = new double[5];
        double bestTestError = DBL_MAX;
        for (int i = 0; i < 5; i++) {
            cout << "**********" << endl;
            cout << "SEED " << seeds[i] << endl;
            cout << "**********" << endl;
            srand(seeds[i]);
            mlp.runBackPropagation(trainDataset, testDataset, maxIter, &(trainErrors[i]), &(testErrors[i]), &(trainCCRs[i]), &(testCCRs[i]), errorFunction);
            cout << "We end!! => Final test CCR: " << testCCRs[i] << endl;

            // We save the weights every time we find a better model
            if (wflag && testErrors[i] <= bestTestError) {
                mlp.saveWeights(weightsFile);
                bestTestError = testErrors[i];
            }
        }

        double trainAverageError = 0, trainStdError = 0;
        double testAverageError = 0, testStdError = 0;
        double trainAverageCCR = 0, trainStdCCR = 0;
        double testAverageCCR = 0, testStdCCR = 0;

        // Obtain training and test averages and standard deviations
        util::getStatistics(trainErrors, 5, trainAverageError, trainStdError);
        util::getStatistics(trainCCRs, 5, trainAverageCCR, trainStdCCR);
        util::getStatistics(testErrors, 5, testAverageError, testStdError);
        util::getStatistics(testCCRs, 5, testAverageCCR, testStdCCR);

        cout << "WE HAVE FINISHED WITH ALL THE SEEDS" << endl;
        cout << "FINAL REPORT" << endl;
        cout << "*************" << endl;
        cout << "Train error (Mean +- SD): " << trainAverageError << " +- " << trainStdError << endl;
        cout << "Test error (Mean +- SD): " << testAverageError << " +- " << testStdError << endl;
        cout << "Train CCR (Mean +- SD): " << trainAverageCCR << " +- " << trainStdCCR << endl;
        cout << "Test CCR (Mean +- SD): " << testAverageCCR << " +- " << testStdCCR << endl;

        delete testDataset;
        delete trainDataset;
        delete[] trainErrors;
        delete[] testErrors;
        delete[] trainCCRs;
        delete[] testCCRs;

        return EXIT_SUCCESS;
    } else {

        //////////////////////////////
        // PREDICTION MODE (KAGGLE) //
        //////////////////////////////

        // You do not have to modify anything from here.

        // Multilayer perceptron object
        MultilayerPerceptron mlp;

        // Initializing the network with the topology vector
        if (!wflag || !mlp.readWeights(weightsFile)) {
            cerr << "Error while reading weights, we can not continue" << endl;
            exit(-1);
        }

        // Reading training and test data: call to mlp.readData(...)
        Dataset* testDataset;
        testDataset = mlp.readData(testFile);
        if (testDataset == NULL) {
            cerr << "The test file is not valid, we can not continue" << endl;
            exit(-1);
        }

        mlp.predict(testDataset);

        delete testDataset;
        return EXIT_SUCCESS;
    }
}
