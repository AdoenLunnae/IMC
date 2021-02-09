#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:14:36 2017

@author: pedroa
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import svm, preprocessing, model_selection


def run(dataset='Datasets/csv/dataset1.csv', c=1000, fig_name="1"):

    # Load the dataset
    data = pd.read_csv(dataset, header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Train the SVM model
    svm_model = svm.SVC(kernel='linear', C=c)
    svm_model.fit(X, y)

    # Plot the points
    plt.figure(fig_name)
    plt.clf()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(fig_name)

    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired)

    # Plot the support vectors class regions, the separating hyperplane and the margins
    plt.axis('tight')
    # |->Plot support vectors
    plt.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1],
                marker='+', s=100, zorder=10, cmap=plt.cm.Paired)
    # |-> Extract the limits
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()
    # |-> Create a grid with all the points and then obtain the SVM
    #    score for all the points
    XX, YY = np.mgrid[x_min:x_max:500j, y_min:y_max:500j]
    Z = svm_model.decision_function(np.c_[XX.ravel(), YY.ravel()])
    # |-> Plot the results in a countour
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, shading='nearest')
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                linestyles=['--', '-', '--'], levels=[-1, 0, 1])

    plt.show()


def run_gaussian(dataset, c=1000, fig_name="1", gamma=1000):

    # Load the dataset
    data = pd.read_csv(dataset, header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Train the SVM model
    svm_model = svm.SVC(kernel='rbf', C=c, gamma=gamma)
    svm_model.fit(X, y)

    # Plot the points
    plt.figure(fig_name)
    plt.clf()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(fig_name)

    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired)

    # Plot the support vectors class regions, the separating hyperplane and the margins
    plt.axis('tight')
    # |-> Extract the limits
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()
    # |-> Create a grid with all the points and then obtain the SVM
    #    score for all the points
    XX, YY = np.mgrid[x_min:x_max:500j, y_min:y_max:500j]
    Z = svm_model.decision_function(np.c_[XX.ravel(), YY.ravel()])
    # |-> Plot the results in a countour
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, shading='nearest')

    plt.show()


def show_data(dataset, fig_name=None):
    # Load the dataset
    data = pd.read_csv(dataset, header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Plot the points
    plt.figure(fig_name or dataset)
    plt.clf()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(fig_name)

    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired)


def split_and_train(dataset, seed=1):
    data = pd.read_csv(dataset, header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.25, random_state=seed)

    scaler = preprocessing.StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    configs = [(c_exp, g_exp) for c_exp in range(-2, 5)
               for g_exp in range(-2, 5)]
    acc_linear = []
    acc_gaussian = []
    acc_linear_train = []
    acc_gaussian_train = []
    for c_exp, g_exp in configs:
        linear_model = svm.SVC(kernel='linear', C=10**c_exp)
        rbf_model = svm.SVC(C=10**c_exp, gamma=10**g_exp)

        linear_model.fit(X_train, y_train)
        rbf_model.fit(X_train, y_train)

        acc_linear.append(linear_model.score(X_test, y_test))
        acc_linear_train.append(linear_model.score(X_train, y_train))

        acc_gaussian.append(rbf_model.score(X_test, y_test))
        acc_gaussian_train.append(rbf_model.score(X_train, y_train))

    best_linear_config = configs[np.argmax(acc_linear)]
    best_linear_score = acc_linear[np.argmax(acc_linear)]
    best_linear_score_train = acc_linear_train[np.argmax(acc_linear)]

    best_gaussian_config = configs[np.argmax(acc_gaussian)]
    best_gaussian_score = acc_gaussian[np.argmax(acc_gaussian)]
    best_gaussian_score_train = acc_gaussian_train[np.argmax(acc_gaussian)]

    print(f"Best linear configuration: C=10^{best_linear_config[0]} - Score: "
          f"Train->{best_linear_score_train*100}%\t"
          f"Test ->{best_linear_score*100}%")

    print(f"Best gaussian configuration: C=10^{best_gaussian_config[0]}, "
          f"gamma=10^{best_gaussian_config[1]} - Score: "
          f"Train->{best_gaussian_score_train*100}%\t"
          f"Test->{best_gaussian_score*100}%")


def cross_validation_search(
        dataset, test_dataset=None, folds=5, test_size=.25, c_vals=None, gamma_vals=None):

    train_data = pd.read_csv(dataset, header=None)

    if test_dataset is None:
        X = train_data.iloc[:, :-1].values
        y = train_data.iloc[:, -1].values
        X_train, X_test, y_train, y_test = \
            model_selection.train_test_split(X, y, test_size=test_size,
                                             random_state=35)
    else:
        test_data = pd.read_csv(test_dataset, header=None)
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

    scaler = preprocessing.StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm_model = svm.SVC()

    Cs = c_vals if c_vals is not None else np.logspace(-4, 4, num=9, base=10)
    Gs = gamma_vals if gamma_vals is not None else np.logspace(
        -4, 4, num=9, base=10)

    optimal = model_selection.GridSearchCV(
        estimator=svm_model, param_grid=dict(C=Cs, gamma=Gs), cv=folds)

    optimal.fit(X_train, y_train)

    chosen_c = optimal.best_estimator_.C
    chosen_gamma = optimal.best_estimator_.gamma

    print(f"Best estimator: C={chosen_c}, gamma={chosen_gamma}")

    print(
        f"Accuracy: Train->{optimal.score(X_train, y_train)*100}% Test->{optimal.score(X_test, y_test)*100}%")


def cross_validation_search_linear(
        dataset, test_dataset=None, folds=5, test_size=.25, c_vals=None):

    train_data = pd.read_csv(dataset, header=None)

    if test_dataset is None:
        X = train_data.iloc[:, :-1].values
        y = train_data.iloc[:, -1].values
        X_train, X_test, y_train, y_test = \
            model_selection.train_test_split(X, y, test_size=test_size,
                                             random_state=35)
    else:
        test_data = pd.read_csv(test_dataset, header=None)
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

    scaler = preprocessing.StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm_model = svm.SVC()

    Cs = c_vals or np.logspace(-4, 4, num=9, base=10)

    optimal = model_selection.GridSearchCV(
        estimator=svm_model, param_grid=dict(C=Cs, kernel=['linear']), cv=folds)

    optimal.fit(X_train, y_train)

    chosen_c = optimal.best_estimator_.C

    print(f"Best estimator: C={chosen_c}")

    print(
        f"Accuracy: Train->{optimal.score(X_train, y_train)*100}% "
        f"Test->{optimal.score(X_test, y_test)*100}%")
