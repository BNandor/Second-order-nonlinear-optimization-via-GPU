//
// Created by spaceman on 2022. 12. 14..
//

#ifndef PARALLELLBFGS_STATISTICS_CUH
#define PARALLELLBFGS_STATISTICS_CUH
#include <algorithm>
#include <vector>

class Statistics{
public:
    double median(std::vector<double> &v)
    {
        if(v.empty()) {
            return 0.0;
        }
        auto n = v.size() / 2;
        std::nth_element(v.begin(), v.begin()+n, v.end());
        auto med = v[n];
        if(!(v.size() & 1)) { //If the set size is even
            auto max_it = std::max_element(v.begin(), v.begin()+n);
            med = (*max_it + med) / 2.0;
        }
        return med;
    }

    double IQR(std::vector<double> &v) {
        auto const Q1 = v.size() / 4;
        auto const Q2 = v.size() / 2;
        auto const Q3 = v.size()%2!=0 ? Q1 + Q2+1 : Q1 + Q2;
        std::vector<double> vSorted(v);
        std::sort(vSorted.begin(),vSorted.end());
        return vSorted[Q3] - vSorted[Q1];
    }
};
#endif //PARALLELLBFGS_STATISTICS_CUH
