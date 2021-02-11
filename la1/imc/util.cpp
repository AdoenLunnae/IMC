#include <cstdlib> // To establish the seed srand() and generate pseudorandom numbers rand()
#include <cmath>
#include <cstring>
namespace util
{
    int *intRandomVectorWithoutRepeating(const int &max, const int &howMany, int *remaining)
    {
        int total = max + 1;
        int *numbersToBeSelected = new int[total];
        int *selection = new int[howMany];
        // Initialize the list of possible selections
        for (int i = 0; i < total; i++)
            numbersToBeSelected[i] = i;

        for (int i = 0; i < howMany; i++)
        {
            int selectedNumber = rand() % (total - i);

            selection[i] = numbersToBeSelected[selectedNumber];
            numbersToBeSelected[selectedNumber] = numbersToBeSelected[total - i - 1];
        }

        if (remaining != 0x0)
            memcpy(remaining, numbersToBeSelected, (total - howMany) * sizeof(int));

        delete[] numbersToBeSelected;
        return selection;
    };

    double getMean(const double *data, const int &n)
    {
        double result = .0;

        for (int i = 0; i < n; ++i)
            result += data[i];

        return result / n;
    }

    double getStandardDeviation(const double *data, const int &n, const double &mean)
    {
        double result = 0;
        for (int i = 0; i < n; ++i)
            result += pow(data[i], 2);

        result /= n;

        return sqrt(result - pow(mean, 2));
    }

    void getStatistics(const double *data, const int &n, double &mean, double &std)
    {
        mean = getMean(data, n);
        std = getStandardDeviation(data, n, mean);
    }

} // namespace util