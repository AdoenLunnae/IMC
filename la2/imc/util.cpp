/*
 * util.h
 *
 *  Created on: 06/03/2015
 *      Author: pedroa
 */
#include "util.h"
#include <cmath>
#include <cstdlib>
namespace util {
int* integerRandomVectorWithoutRepeating(int min, int max, int howMany, int** remaining)
{
    int total = max - min + 1;
    int* numbersToBeSelected = new int[total];
    int* numbersSelected = new int[howMany];
    // Initialize the list of possible selections
    for (int i = 0; i < total; i++)
        numbersToBeSelected[i] = min + i;

    for (int i = 0; i < howMany; i++) {
        int selectedNumber = rand() % (total - i);
        // Store the selected number
        numbersSelected[i] = numbersToBeSelected[selectedNumber];
        // We include the last valid number in numbersToBeSelected, in this way
        // all numbers are valid until total-i-1
        numbersToBeSelected[selectedNumber] = numbersToBeSelected[total - i - 1];
    }
    if (remaining != 0x0)
        *remaining = numbersToBeSelected;
    else
        delete[] numbersToBeSelected;

    return numbersSelected;
};

double getMean(const double* data, const int& n)
{
    double result = .0;

    for (unsigned int i = 0; i < n; ++i)
        result += data[i];

    return result / n;
}

double getStandardDeviation(const double* data, const int& n, const double& mean)
{
    double result = 0;
    for (unsigned int i = 0; i < n; ++i)
        result += data[i] * data[i];

    result /= n;

    return sqrt(result - (mean * mean));
}

void getStatistics(const double* data, const int& n, double& mean, double& std)
{
    mean = getMean(data, n);
    std = getStandardDeviation(data, n, mean);
}

};
