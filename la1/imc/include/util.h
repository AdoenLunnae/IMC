/*
 * util.h
 *
 *  Created on: 06/03/2015
 *      Author: pedroa
 */

#ifndef UTIL_H_
#define UTIL_H_
#include <cstdlib> // To establish the seed srand() and generate pseudorandom numbers rand()
#include <cmath>
namespace util
{

        static int *intRandomVectorWithoutRepeating(const int &max, const int &howMany)
        {
                int *numbersToBeSelected = new int[max];
                int *selection = new int[howMany];
                // Initialize the list of possible selections
                for (int i = 0; i < max; i++)
                {
                        numbersToBeSelected[i] = i;
                }

                for (int i = 0; i < howMany; i++)
                {
                        int selectedNumber = rand() % (max - i);

                        selection[i] = numbersToBeSelected[selectedNumber];
                        numbersToBeSelected[selectedNumber] = numbersToBeSelected[max - i - 1];
                }

                delete[] numbersToBeSelected;
                return selection;
        };

        static bool *boolRandomVector(const int &size, const int &howManyTrue)
        {
                int *indexes = intRandomVectorWithoutRepeating(size, howManyTrue);
                bool *selection = new bool[size];

                for (int i = 0; i < size; ++i)
                        selection[i] = false;

                for (int i = 0; i < howManyTrue; ++i)
                        selection[indexes[i]] = true;

                return selection;
        };

        static double getMean(const double *data, const int &n)
        {
                double result = .0;

                for (unsigned int i = 0; i < n; ++i)
                        result += data[i];

                return result / n;
        }

        static double getStandardDeviation(const double *data, const int &n, const double &mean)
        {
                double result = 0;
                for (unsigned int i = 0; i < n; ++i)
                        result += data[i] * data[i];

                result /= n;

                return sqrt(result - (mean * mean));
        }

        static void getStatistics(const double *data, const int &n, double &mean, double &std)
        {
                mean = getMean(data, n);
                std = getStandardDeviation(data, n, mean);
        }

} // namespace util

#endif /* UTIL_H_ */
