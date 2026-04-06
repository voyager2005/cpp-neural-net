#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>

/*
The original documentation for MNIST has the data saved in big endian, 
will have to convert this into little endian for Intel Processor
*/ 
uint32_t reverse_bytes(uint32_t c) {
    return ((c >> 24) & 0xff) | 
           ((c << 8) & 0xff0000) | 
           ((c >> 8) & 0xff00) | 
           ((c << 24) & 0xff000000);
}

// image loading and normalization (truth be told, some of this is off stack overflow, loading images on cpp is very painful)
std::vector<Eigen::VectorXf> read_mnist_images(const std::string& full_path) {
    std::vector<Eigen::VectorXf> dataset;
    std::ifstream file(full_path, std::ios::binary);

    // "try" "catch" using if else (should modify this to be a try catch if that exists in cpp)
    if (file.is_open()) {
        uint32_t magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
        
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&num_images, sizeof(num_images));
        file.read((char*)&num_rows, sizeof(num_rows));
        file.read((char*)&num_cols, sizeof(num_cols));

        magic_number = reverse_bytes(magic_number);
        num_images = reverse_bytes(num_images);
        num_rows = reverse_bytes(num_rows);
        num_cols = reverse_bytes(num_cols);

        int image_size = num_rows * num_cols; // 28 x 28 = 784

        // Loop through all images
        for (int i = 0; i < num_images; ++i) {
            Eigen::VectorXf image_vec(image_size);
            
            // Loop through all pixels in the current image
            for (int j = 0; j < image_size; ++j) {
                unsigned char pixel = 0;
                file.read((char*)&pixel, sizeof(pixel));
                image_vec(j) = (float)pixel / 255.0f; // Normalize
            }
            dataset.push_back(image_vec);
        }
        file.close();
    } else {
        std::cout << "Error: Could not open file " << full_path << "!\n";
    }
    return dataset;
}

// Loads the corresponding integer labels (0 through 9)
std::vector<int> read_mnist_labels(const std::string& full_path) {
    std::vector<int> labels;
    std::ifstream file(full_path, std::ios::binary);

    if (file.is_open()) {
        uint32_t magic_number = 0, num_items = 0;
        
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&num_items, sizeof(num_items));

        magic_number = reverse_bytes(magic_number);
        num_items = reverse_bytes(num_items);

        for (int i = 0; i < num_items; ++i) {
            unsigned char label = 0;
            file.read((char*)&label, sizeof(label));
            labels.push_back((int)label); // Store as standard integer
        }
        file.close();
    }
    return labels;
}