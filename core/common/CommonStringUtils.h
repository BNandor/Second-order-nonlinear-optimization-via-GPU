//
// Created by spaceman on 2023. 06. 03..
//

#ifndef PARALLELLBFGS_COMMONSTRINGUTILS_H
#define PARALLELLBFGS_COMMONSTRINGUTILS_H
#include <iostream>
#include <sstream>
#include <set>
#include <string>
namespace stringutil {
    std::set <std::string> splitString(const std::string &str, char delimiter) {
        std::set <std::string> result;
        std::stringstream ss(str);
        std::string token;

        while (std::getline(ss, token, delimiter)) {
            result.insert(token);
        }

        return result;
    }
}
#endif //PARALLELLBFGS_COMMONSTRINGUTILS_H
