//
// Created by spaceman on 2022. 12. 19..
//

#ifndef PARALLELLBFGS_LOGS_CUH
#define PARALLELLBFGS_LOGS_CUH
#include <json.hpp>
#include <iostream>
#include <fstream>
using json = nlohmann::json;

class Logs {
public:
    static void appendLogs(json& experimentLogs, std::string logsFile) {
        std::fstream currentLogsFile;
        currentLogsFile.open(logsFile.c_str());
        json currentLogs;
        if(currentLogsFile.is_open()) {
            std::cout<<"Parsing "<<logsFile;
            currentLogs=json::parse(currentLogsFile);
            std::cout<<"\nparsed\n"<<currentLogs<<std::endl;
            currentLogs["experiments"].push_back(experimentLogs);
            std::cout<<"\nappended\n"<<currentLogs<<std::endl;
            currentLogsFile.close();
            std::ofstream outFile;
            outFile.open(logsFile.c_str());
            outFile<<currentLogs;
            outFile.close();
        }else{
            std::ofstream outFile;
            outFile.open(logsFile.c_str());
            std::cout<<"Creating "<<logsFile;
            if(!outFile.is_open()) {
                std::cerr<<"Could not create "<<logsFile<<std::endl;
                std::exit(2);
            }
            currentLogs["experiments"].push_back(experimentLogs);
            outFile<<currentLogs;
            outFile.close();
        }
    }
};
#endif //PARALLELLBFGS_LOGS_CUH
