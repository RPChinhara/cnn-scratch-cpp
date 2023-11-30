#include "datasets.h"

#include <fstream>
#include <sstream>
#include <vector>

Tensor load_air_passengers() {
    std::ifstream file("datasets\\air_passengers.csv");

    if (!file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;

    // Skip the first line.
    std::getline(file, line);

    int idx = 0;
    Tensor dataset = Tensor({ 0.0 }, { 144, 1 });

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        
        // Skip the first column which is date.
        std::getline(ss, value, ',');

        std::getline(ss, value, ',');
        dataset[idx] = std::stof(value);
        ++idx;
    }

    file.close();

    return dataset;
}

Cifar10 load_cifar10() {

     // Open the binary data file.
    std::ifstream dataFile("datasets\\cifar10\\data_batch_1.bin", std::ios::binary);

    if (!dataFile.is_open()) {
        std::cerr << "Error opening data file." << std::endl;
        exit(1);
    }

    // Create an array to store labels.
    std::string labels[10] = {
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
    };

    // Read label and image data for proof.
    uint8_t labelValue;
    dataFile.read(reinterpret_cast<char*>(&labelValue), sizeof(labelValue));

    // Read image data (assuming a single image size is 3072 bytes)
    std::vector<uint8_t> imageData(3072);
    dataFile.read(reinterpret_cast<char*>(imageData.data()), 3072);

    // Close the file.
    dataFile.close();

    // Print label and a portion of the image data for proof.
    std::cout << "Label: " << labels[labelValue] << std::endl;
    std::cout << "Image Data (First 10 Bytes): ";
    for (int i = 0; i < 10; ++i) {
        std::cout << static_cast<int>(imageData[i]) << " ";
    }
    std::cout << std::endl;

    return Cifar10();
}

Imdb load_imdb() {
    std::ifstream file("datasets\\imdb.csv");

    if (!file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    std::getline(file, line);
    std::stringstream ss(line);
    std::string value;
    
    // Skip the first column which is an ID.
    std::getline(ss, value, ',');
    std::cout << value << std::endl;

    std::getline(ss, value, ',');
    std::cout << value << std::endl;
    return Imdb();
}

Iris load_iris() {
    std::ifstream file("datasets\\iris.csv");

    if (!file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;

    // Skip the first line.
    std::getline(file, line);

    int idx_features = 0;
    int idx_target   = 0;
    Tensor features = Tensor({ 0.0 }, { 150, 4 });
    Tensor target   = Tensor({ 0.0 }, { 150, 1 });

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        
        // Skip the first column which is an ID.
        std::getline(ss, value, ',');

        std::getline(ss, value, ',');
        features[idx_features] = std::stof(value);
        ++idx_features;
        
        std::getline(ss, value, ',');
        features[idx_features] = std::stof(value);
        ++idx_features;
        
        std::getline(ss, value, ',');
        features[idx_features] = std::stof(value);
        ++idx_features;

        std::getline(ss, value, ',');
        features[idx_features] = std::stof(value);
        ++idx_features;

        std::getline(ss, value);
        if (value == "Iris-setosa") {
            target[idx_target] = 0.0f;
            ++idx_target;
        } else if (value == "Iris-versicolor") {
            target[idx_target] = 1.0f;
            ++idx_target;
        } else if (value == "Iris-virginica") {
            target[idx_target] = 2.0f;
            ++idx_target;
        }
    }
    
    file.close();

    Iris iris;
    iris.features = features;
    iris.target   = target;

    return iris;
}

Mnist load_mnist() {
    return Mnist();
}