//
// Created by spaceman on 2024. 03. 23..
//

#ifndef PARALLELLBFGS_SPRTTTEST_H
#define PARALLELLBFGS_SPRTTTEST_H
#include <boost/math/distributions/non_central_t.hpp>

class SPRTTTest {
public:
    static std::pair<double, double> sprtTTestBoundsAB(double alpha, double beta) {
        return {(1.0 - beta) / alpha, beta / (1.0 - alpha)};
    }

    static double pooledStandardDeviation(const std::vector<double>& samples1, const std::vector<double>& samples2) {
        int n1 = samples1.size();
        int n2 = samples2.size();
        if (n1 < 2 || n2 < 2) {
            std::cerr << "invalid sample sizes: " << n1 << " and " << n2 << std::endl;
            return NAN;
        }
        double variance1 = 0.0;
        double mean1=mean(samples1);
        double mean2=mean(samples2);
        for (int i = 0; i < n1; i++) {
            variance1 += (samples1[i] - mean1) * (samples1[i] - mean1);
        }
        variance1 /= (n1 - 1);
        double variance2 = 0.0;
        for (int i = 0; i < n2; i++) {
            variance2 += (samples2[i] - mean2) * (samples2[i] - mean2);
        }
        variance2 /= (n2 - 1);
        return std::sqrt(((n1 - 1) * variance1 + (n2 - 1) * variance2) / (n1 + n2 - 2));
    }

    static double mean(const std::vector<double>& samples) {
        double sum = 0.0;
        for (int i = 0; i < samples.size(); i++) {
            sum += samples[i];
        }
        return sum / samples.size();
    }

    static double t(const std::vector<double>& samples1, const std::vector<double>& samples2) {
        int n1 = samples1.size();
        int n2 = samples2.size();
        if (n1 < 2 || n2 < 2) {
            std::cerr << "invalid sample sizes: " << n1 << " and " << n2 << std::endl;
            return NAN;
        }
        return (mean(samples1) - mean(samples2)) / (pooledStandardDeviation(samples1, samples2) * std::sqrt(1.0 / n1 + 1.0 / n2));
    }

    static double oneSidedTRatio(const std::vector<double>& samples1, const std::vector<double>& samples2, double cohensD) {
        int n1 = samples1.size();
        int n2 = samples2.size();
        double delta = cohensD * std::sqrt((n1 * n2) / (n1 + n2));
        double tstat = t(samples1, samples2);
        int df = n1 + n2 - 2;
        boost::math::non_central_t_distribution<> nonCentralTH1(df, delta);
        boost::math::non_central_t_distribution<> nonCentralTH0(df, 0);
        return boost::math::pdf(nonCentralTH1, tstat) / boost::math::pdf(nonCentralTH0, tstat);
    }

    static int checkIfNewMeanIsLessThanComparisonSequentialT(const std::vector<double>& samples1, const std::vector<double>& samples2, double alpha, double beta, double cohensD) {
        std::pair<double, double> bounds = sprtTTestBoundsAB(alpha, beta);
        double A=bounds.first;
        double B=bounds.second;
        double R = oneSidedTRatio(samples1, samples2, cohensD);
        std::cout << "B,A = " << B << "<->" << A<<"R: "<<R<<std::endl;
        if (R >= A){
            return 1;
        }
        if (R <= B) {
            return 0;
        }
        return -1;
    }
};

#endif //PARALLELLBFGS_SPRTTTEST_H