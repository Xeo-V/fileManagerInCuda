#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <unordered_map>
#include <list>
#include <vector>

std::unordered_map<std::string, std::string> fileSystem;
std::unordered_map<std::string, std::list<std::string>::iterator> cacheMap;
std::list<std::string> lruList;

__global__ void stringSearchKernel(const char* text, const char* query, int* positions, int textSize, int querySize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx + querySize > textSize) return;

    bool match = true;
    for (int i = 0; i < querySize; ++i) {
        if (text[idx + i] != query[i]) {
            match = false;
            break;
        }
    }
    positions[idx] = (match) ? 1 : 0;
}

void initializeFileSystem() {
    fileSystem.clear();
    cacheMap.clear();
    lruList.clear();
}

bool createFile(const std::string& fileName) {
    if (fileSystem.find(fileName) != fileSystem.end()) {
        return false;
    }
    fileSystem[fileName] = "";
    return true;
}

bool writeToFile(const std::string& fileName, const std::string& data) {
    if (fileSystem.find(fileName) == fileSystem.end()) {
        return false;
    }
    fileSystem[fileName] = data;
    return true;
}

bool readFromFile(const std::string& fileName, std::string& data) {
    if (fileSystem.find(fileName) == fileSystem.end()) {
        return false;
    }

    if (cacheMap.find(fileName) != cacheMap.end()) {
        lruList.erase(cacheMap[fileName]);
        lruList.push_front(fileName);
        cacheMap[fileName] = lruList.begin();
        data = fileSystem[fileName];
    }
    else {
        if (lruList.size() == 5) {
            std::string last = lruList.back();
            lruList.pop_back();
            cacheMap.erase(last);
        }

        lruList.push_front(fileName);
        cacheMap[fileName] = lruList.begin();
        data = fileSystem[fileName];
    }

    return true;
}

bool searchStringInFile(const std::string& fileName, const std::string& query, std::vector<int>& positions) {
    std::string fileContent;
    if (!readFromFile(fileName, fileContent)) {
        return false;
    }

    int* d_positions;
    char* d_text, * d_query;
    int textSize = fileContent.size();
    int querySize = query.size();
    positions.resize(textSize, 0);

    cudaMalloc(&d_positions, textSize * sizeof(int));
    cudaMalloc(&d_text, textSize * sizeof(char));
    cudaMalloc(&d_query, querySize * sizeof(char));

    cudaMemcpy(d_text, fileContent.c_str(), textSize * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query.c_str(), querySize * sizeof(char), cudaMemcpyHostToDevice);

    stringSearchKernel << <(textSize + 255) / 256, 256 >> > (d_text, d_query, d_positions, textSize, querySize);

    cudaMemcpy(positions.data(), d_positions, textSize * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_positions);
    cudaFree(d_text);
    cudaFree(d_query);

    return true;
}

int main() {
    initializeFileSystem();

    while (true) {
        std::cout << "\nChoose an operation:\n";
        std::cout << "1. Create file\n";
        std::cout << "2. Write to file\n";
        std::cout << "3. Read from file\n";
        std::cout << "4. List files\n";
        std::cout << "5. Search string in file\n";
        std::cout << "6. Exit\n";

        int choice;
        std::cin >> choice;

        std::string fileName, data, query;
        std::vector<int> positions;
        switch (choice) {
        case 1:
            std::cout << "Enter the file name: ";
            std::cin >> fileName;
            createFile(fileName);
            break;
        case 2:
            std::cout << "Enter the file name: ";
            std::cin >> fileName;
            std::cout << "Enter the data: ";
            std::cin.ignore();
            std::getline(std::cin, data);
            writeToFile(fileName, data);
            break;
        case 3:
            std::cout << "Enter the file name: ";
            std::cin >> fileName;
            if (readFromFile(fileName, data)) {
                std::cout << "Data in " << fileName << ": " << data << '\n';
            }
            break;
        case 4:
            for (const auto& entry : fileSystem) {
                std::cout << entry.first << '\n';
            }
            break;
        case 5:
            std::cout << "Enter the file name: ";
            std::cin >> fileName;
            std::cout << "Enter the search query: ";
            std::cin >> query;
            if (searchStringInFile(fileName, query, positions)) {
                std::cout << "Positions found: ";
                for (int i = 0; i < positions.size(); ++i) {
                    if (positions[i] == 1) {
                        std::cout << i << ' ';
                    }
                }
                std::cout << '\n';
            }
            break;
        case 6:
            return 0;
        default:
            std::cout << "Invalid choice!\n";
            break;
        }
    }

    return 0;
}
