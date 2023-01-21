//
// Created by spaceman on 2022. 12. 19..
//

#ifndef PARALLELLBFGS_JSONOPERATIONS_CUH
#define PARALLELLBFGS_JSONOPERATIONS_CUH
#include <json.hpp>
#include <iostream>
#include <fstream>
using json = nlohmann::json;

class JsonOperations {
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

    static json loadJsonFrom(std::string jsonPath){
        std::fstream file;
        file.open(jsonPath.c_str());
        json jsonContent;
        if(file.is_open()) {
            std::cout << "Parsing " << jsonPath;
            jsonContent = json::parse(file);
            std::cout << "\nparsed\n" << jsonContent << std::endl;
            file.close();
        }
        else{
            std::cerr<<"Could not open "<<jsonPath<<std::endl;
            exit(2);
        }
        return jsonContent;
    }
};
#endif //PARALLELLBFGS_JSONOPERATIONS_CUH
