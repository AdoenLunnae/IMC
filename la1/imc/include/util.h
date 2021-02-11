/*
 * util.h
 *
 *  Created on: 06/03/2015
 *      Author: pedroa
 */

#ifndef UTIL_H_
#define UTIL_H_
namespace util
{

        int *intRandomVectorWithoutRepeating(const int &max, const int &howMany, int *remaining=0x0);
        
        double getMean(const double *data, const int &n);
        double getStandardDeviation(const double *data, const int &n, const double &mean);
        void getStatistics(const double *data, const int &n, double &mean, double &std);

} // namespace util

#endif /* UTIL_H_ */
