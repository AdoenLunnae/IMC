/*
 * util.h
 *
 *  Created on: 06/03/2015
 *      Author: pedroa
 */

#ifndef UTIL_H_
#define UTIL_H_
#include <cmath>
#include <cstdlib> // To establish the seed srand() and generate pseudorandom numbers rand()
namespace util {
static int* integerRandomVectoWithoutRepeating(int min, int max, int howMany)
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
    delete[] numbersToBeSelected;
    return numbersSelected;
};

static double getAverage(const double* data, const int& n)
{
    double result = .0;

    for (unsigned int i = 0; i < n; ++i)
        result += data[i];

    return result / n;
}

static double getMean(const double* data, const int& n)
{
    double result = .0;

    for (unsigned int i = 0; i < n; ++i)
        result += data[i];

    return result / n;
}

static double getStandardDeviation(const double* data, const int& n, const double& mean)
{
    double result = 0;
    for (unsigned int i = 0; i < n; ++i)
        result += data[i] * data[i];

    result /= n;

    return sqrt(result - (mean * mean));
}

static void getStatistics(const double* data, const int& n, double& mean, double& std)
{
    mean = getMean(data, n);
    std = getStandardDeviation(data, n, mean);
}

};
#endif /* UTIL_H_ */
