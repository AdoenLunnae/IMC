#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 12:37:04 2020

IMC: lab assignment 3

@author: pagutierrez
"""

import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
import sklearn.cluster
import sklearn.linear_model
import pickle
import os
import click


def train_rbf_total(train_file, test_file, classification, ratio_rbf, l2, eta, outputs, model, pred):
    """ 5 executions of RBFNN training

        RBF neural network based on hybrid supervised/unsupervised training.
        We run 5 executions with different seeds.
    """

    if not pred:

        if train_file is None:
            print("You have not specified the training file (-t)")
            return

        train_mses = np.empty(1)
        train_ccrs = np.empty(1)
        test_mses = np.empty(1)
        test_ccrs = np.empty(1)

        s = 3
        print("-----------")
        print("Seed: %d" % s)
        print("-----------")
        np.random.seed(s)
        train_mses[0], test_mses[0], train_ccrs[0], test_ccrs[0] = \
            train_rbf(train_file, test_file, classification, ratio_rbf, l2, eta, outputs,
                      model and "{}/{}.pickle".format(model, s) or "")

        print("******************")
        print("Summary of results")
        print("******************")
        print("Training MSE: %f +- %f" %
              (np.mean(train_mses), np.std(train_mses)))
        print("Test MSE: %f +- %f" % (np.mean(test_mses), np.std(test_mses)))
        print("Training CCR: %.2f%% +- %.2f%%" %
              (np.mean(train_ccrs), np.std(train_ccrs)))
        print("Test CCR: %.2f%% +- %.2f%%" %
              (np.mean(test_ccrs), np.std(test_ccrs)))

    else:
        # KAGGLE
        if model is None:
            print("You have not specified the file with the model (-m).")
            return

        # Obtain the predictions for the test set
        predictions = predict(test_file, model)

        # Print the predictions in csv format
        print("Id,Category")
        for index, prediction in enumerate(predictions):
            s = ""
            s += str(index)
            if isinstance(prediction, np.ndarray):
                for output in prediction:
                    s += ",{}".format(output)
            else:
                s += ",{}".format(int(prediction))

            print(s)


def train_rbf(train_file, test_file, classification, ratio_rbf, l2, eta, outputs, model_file=""):
    """ One execution of RBFNN training

        RBF neural network based on hybrid supervised/unsupervised training.
        We run 1 executions.

        Parameters
        ----------
        train_file: string
            Name of the training file
        test_file: string
            Name of the test file
        classification: bool
            True if it is a classification problem
        ratio_rbf: float
            Ratio (as a fraction of 1) indicating the number of RBFs
            with respect to the total number of patterns
        l2: bool
            True if we want to use L2 regularization for logistic regression
            False if we want to use L1 regularization for logistic regression
        eta: float
            Value of the regularization factor for logistic regression
        outputs: int
            Number of variables that will be used as outputs (all at the end
            of the matrix)
        model_file: string
            Name of the file where the model will be written

        Returns
        -------
        train_mse: float
            Mean Squared Error for training data
            For classification, we will use the MSE of the predicted probabilities
            with respect to the target ones (1-of-J coding)
        test_mse: float
            Mean Squared Error for test data
            For classification, we will use the MSE of the predicted probabilities
            with respect to the target ones (1-of-J coding)
        train_ccr: float
            Training accuracy (CCR) of the model
            For regression, we will return a 0
        test_ccr: float
            Training accuracy (CCR) of the model
            For regression, we will return a 0
    """
    train_inputs, train_outputs, test_inputs, test_outputs = read_data(train_file,
                                                                       test_file,
                                                                       outputs)

    # Obtain num_rbf from ratio_rbf
    n_patterns = train_inputs.shape[0]
    num_rbf = int(round(n_patterns * ratio_rbf))
    print("Number of RBFs used: %d" % (num_rbf))
    kmeans, distances, centers = clustering(classification, train_inputs,
                                            train_outputs, num_rbf)

    radii = calculate_radii(centers, num_rbf)

    r_matrix = calculate_r_matrix(distances, radii)

    if classification:
        logreg = logreg_classification(r_matrix, train_outputs, l2, eta)
    else:
        coefficients = invert_matrix_regression(r_matrix, train_outputs)
    """
    Obtain the distances from the centroids to the test patterns
    and obtain the R matrix for the test set
    """
    test_distances = kmeans.transform(test_inputs)

    test_r_matrix = calculate_r_matrix(test_distances, radii)

    # # # # KAGGLE # # # #
    if model_file != "":
        save_obj = {
            'classification': classification,
            'radii': radii,
            'kmeans': kmeans
        }
        if not classification:
            save_obj['coefficients'] = coefficients
        else:
            save_obj['logreg'] = logreg

        dir = os.path.dirname(model_file)
        if not os.path.isdir(dir):
            os.makedirs(dir)

        with open(model_file, 'wb') as f:
            pickle.dump(save_obj, f)

    # # # # # # # # # # #

    if classification:
        """
        TODO: Obtain the predictions for training and test and calculate
              the CCR. Obtain also the MSE, but comparing the obtained
              probabilities and the target probabilities
        """
        train_predictions = logreg.predict(r_matrix)
        test_predictions = logreg.predict(test_r_matrix)
        train_ccr = sklearn.metrics.accuracy_score(
            train_outputs, train_predictions) * 100
        test_ccr = sklearn.metrics.accuracy_score(
            test_outputs, test_predictions) * 100
    else:
        """
        TODO: Obtain the predictions for training and test and calculate
              the MSE
        """
        train_predictions = np.dot(r_matrix, coefficients)
        test_predictions = np.dot(test_r_matrix, coefficients)
        train_ccr, test_ccr = 0, 0
    train_mse = sklearn.metrics.mean_squared_error(
        train_outputs, train_predictions)
    test_mse = sklearn.metrics.mean_squared_error(
        test_outputs, test_predictions)
    return train_mse, test_mse, train_ccr, test_ccr


def read_data(train_file, test_file, outputs):
    """ Read the input data
        It receives the name of the input data file names (training and test)
        and returns the corresponding matrices

        Parameters
        ----------
        train_file: string
            Name of the training file
        test_file: string
            Name of the test file
        outputs: int
            Number of variables to be used as outputs
            (all at the end of the matrix).

        Returns
        -------
        train_inputs: array, shape (n_train_patterns,n_inputs)
            Matrix containing the inputs for the training patterns
        train_outputs: array, shape (n_train_patterns,n_outputs)
            Matrix containing the outputs for the training patterns
        test_inputs: array, shape (n_test_patterns,n_inputs)
            Matrix containing the inputs for the test patterns
        test_outputs: array, shape (n_test_patterns,n_outputs)
            Matrix containing the outputs for the test patterns
    """
    def dataset_from_file(file, outputs):
        data = pd.read_csv(file, header=None)
        n_columns = data.shape[1]
        data_array = data.to_numpy()

        return (data_array[:, 0:n_columns - outputs], data_array[:, n_columns - outputs:])

    return (
        *dataset_from_file(train_file, outputs),
        *dataset_from_file(test_file, outputs)
    )


def init_centroids_classification(train_inputs, train_outputs, num_rbf):
    """ Initialize the centroids for the case of classification
        This method selects, approximately, num_rbf/num_clases
        patterns for each class.

        Parameters
        ----------
        train_inputs: array, shape (n_patterns,n_inputs)
            Matrix with all the input variables
        train_outputs: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset
        num_rbf: int
            Number of RBFs to be used in the network

        Returns
        -------
        centroids: array, shape (num_rbf,n_inputs)
            Matrix with all the centroids already selected
    """
    split = sklearn.model_selection.train_test_split(
        train_inputs, test_size=num_rbf, stratify=train_outputs)

    return np.asarray(split[1])


def clustering(classification, train_inputs, train_outputs, num_rbf):
    """ It applies the clustering process
        A clustering process is applied to set the centers of the RBFs.
        In the case of classification, the initial centroids are set
        using the method init_centroids_classification().
        In the case of regression, the centroids have to be set randomly.

        Parameters
        ----------
        classification: bool
            True if it is a classification problem
        train_inputs: array, shape (n_patterns,n_inputs)
            Matrix with all the input variables
        train_outputs: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset
        num_rbf: int
            Number of RBFs to be used in the network

        Returns
        -------
        kmeans: sklearn.cluster.KMeans
            KMeans object after the clustering
        distances: array, shape (n_patterns,num_rbf)
            Matrix with the distance from each pattern to each RBF center
        centers: array, shape (num_rbf,n_inputs)
            Centers after the clustering
    """
    init = (init_centroids_classification(train_inputs, train_outputs, num_rbf)
            if classification else 'random')

    n_init = (1 if classification else 10)

    kmeans = sklearn.cluster.KMeans(
        n_clusters=num_rbf, init=init, n_init=n_init)

    kmeans.fit(train_inputs, y=train_outputs)

    centers = kmeans.cluster_centers_
    distances = kmeans.transform(train_inputs)

    return kmeans, distances, centers


def calculate_radii(centers, num_rbf):
    """ It obtains the value of the radii after clustering
        This methods is used to heuristically obtain the radii of the RBFs
        based on the centers

        Parameters
        ----------
        centers: array, shape (num_rbf,n_inputs)
            Centers from which obtain the radii
        num_rbf: int
            Number of RBFs to be used in the network

        Returns
        -------
        radii: array, shape (num_rbf,)
            Array with the radius of each RBF
    """
    radii = np.empty(num_rbf)

    for c_index in range(num_rbf):
        sum_distances = sum(np.sqrt(np.sum((centers[c_index] - centers[other_c_index])**2))
                            for other_c_index in range(num_rbf)
                            if c_index != other_c_index)
        radii[c_index] = sum_distances / (2 * (num_rbf-1))

    return radii


def calculate_r_matrix(distances, radii):
    """ It obtains the R matrix
        This method obtains the R matrix (as explained in the slides),
        which contains the activation of each RBF for each pattern, including
        a final column with ones, to simulate bias

        Parameters
        ----------
        distances: array, shape (n_patterns,num_rbf)
            Matrix with the distance from each pattern to each RBF center
        radii: array, shape (num_rbf,)
            Array with the radius of each RBF

        Returns
        -------
        r_matrix: array, shape (n_patterns,num_rbf+1)
            Matrix with the activation of every RBF for every pattern. Moreover
            we include a last column with ones, which is going to act as bias
    """
    def rbf_out(distance, radius):
        return np.exp(-0.5*np.square(distance/radius))

    n_patterns = distances.shape[0]
    num_rbf = distances.shape[1]

    r_matrix = np.empty((n_patterns, num_rbf + 1))

    for p_index in range(n_patterns):
        for n_index in range(num_rbf):
            r_matrix[p_index, n_index] = rbf_out(
                distances[p_index, n_index], radii[n_index])
        r_matrix[p_index, num_rbf] = 1
    return r_matrix


def invert_matrix_regression(r_matrix: np.ndarray, train_outputs):
    """ Inversion of the matrix for regression case
        This method obtains the pseudoinverse of the r matrix and multiplies
        it by the targets to obtain the coefficients in the case of linear
        regression

        Parameters
        ----------
        r_matrix: array, shape (n_patterns,num_rbf+1)
            Matrix with the activation of every RBF for every pattern. Moreover
            we include a last column with ones, which is going to act as bias
        train_outputs: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset

        Returns
        -------
        coefficients: array, shape (n_outputs,num_rbf+1)
            For every output, values of the coefficients for each RBF and value
            of the bias
    """
    pseudoinverse = np.linalg.pinv(r_matrix)

    n_outputs = train_outputs.shape[1]
    num_rbf_plus_1 = r_matrix.shape[1]

    coefficients = np.empty((n_outputs, num_rbf_plus_1))

    coefficients = pseudoinverse @ train_outputs

    return coefficients


def logreg_classification(r_matrix, train_outputs, l2, eta):
    """ Performs logistic regression training for the classification case
        It trains a logistic regression object to perform classification based
        on the R matrix (activations of the RBFs together with the bias)

        Parameters
        ----------
        r_matrix: array, shape (n_patterns,num_rbf+1)
            Matrix with the activation of every RBF for every pattern. Moreover
            we include a last column with ones, which is going to act as bias
        train_outputs: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset
        l2: bool
            True if we want to use L2 regularization for logistic regression
            False if we want to use L1 regularization for logistic regression
        eta: float
            Value of the regularization factor for logistic regression

        Returns
        -------
        logreg: sklearn.linear_model.LogisticRegression
            Scikit-learn logistic regression model already trained
    """
    norm = 'l2' if l2 else 'l1'

    logreg = sklearn.linear_model.LogisticRegression(
        penalty=norm, C=1/eta, solver='liblinear')
    logreg.fit(r_matrix, y=train_outputs[:, 0])
    return logreg


def predict(test_file, model_file):
    """ Performs a prediction with RBFNN model
        It obtains the predictions of a RBFNN model for a test file, using two files, one
        with the test data and one with the model

        Parameters
        ----------
        test_file: string
            Name of the test file
        model_file: string
            Name of the file containing the model data

        Returns
        -------
        test_predictions: array, shape (n_test_patterns,n_outputs)
            Predictions obtained with the model and the test file inputs
    """
    test_df = pd.read_csv(test_file, header=None)
    test_inputs = test_df.values[:, :]

    with open(model_file, 'rb') as f:
        saved_data = pickle.load(f)

    import ipdb
    ipdb.set_trace()
    radii = saved_data['radii']
    classification = saved_data['classification']
    kmeans = saved_data['kmeans']

    test_distancias = kmeans.transform(test_inputs)
    test_r = calculate_r_matrix(test_distancias, radii)

    if classification:
        logreg = saved_data['logreg']
        return logreg.predict(test_r)
    else:
        coeficientes = saved_data['coefficients']
        return np.dot(test_r, coeficientes)


if __name__ == "__main__":
    train_rbf_total()
