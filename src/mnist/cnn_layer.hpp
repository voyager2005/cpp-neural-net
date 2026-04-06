#pragma once
#include <iostream>
#include <Eigen/Dense>

/**
author: Akshat G

I have personally commented all the sections because in the future I want to implement the following
-> CPP code for Adam Optimizer
-> model checkpoints
-> batch normalization

Documenting this current code so that I can start where i left off. 
Documenting Date: 7th April 2026, 3:10am (IST

Here is a counter to the number of attemps on adding the following features: 
counter: 0



/**
 * @class Conv2D
 * 
 * @brief A 2d conv layer with 3x3 filer and online learning for backprop. 
 * The Conv2D class performs feature extraction by sliding a 3x3 kernel over the input matrix
 * During the forward pass, it catches the input image. During the backward pass, it computes 
 * gradients for the filter and bias. 
 * After the computation, ive added code to apply Stochastic Gradient Descent.
 * it calculates the error gradient to pass backward to the preceding layer using reverse 
 * sliding window accumulation technique.
 */

class Conv2D {
private:
    Eigen::MatrixXf filter; // 3x3 kernel mentioned
    float bias;
    Eigen::MatrixXf last_input; // CACHE: We need the original image for dW

public:
    Conv2D() {
        // weights are initialized to a smaller value to prevent exploding gradients down the line
        filter = Eigen::MatrixXf::Random(3, 3) * 0.1f; // should experiment with other scales
        bias = 0.0f;
    }

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) {
        last_input = input; // catching input here (NOTE for myself)
        
        int filter_size = filter.rows(); 
        int out_size = input.rows() - filter_size + 1; 
        
        Eigen::MatrixXf output = Eigen::MatrixXf::Zero(out_size, out_size);

        for (int i = 0; i < out_size; ++i) {
            for (int j = 0; j < out_size; ++j) {
                Eigen::MatrixXf patch = input.block(i, j, filter_size, filter_size);
                output(i, j) = (patch.cwiseProduct(filter)).sum() + bias;
            }
        }
        return output;
    }

    // Conv2D Back propagation phase
    Eigen::MatrixXf backward(const Eigen::MatrixXf& dZ, float learning_rate) {
        int filter_size = filter.rows();
        int out_size = dZ.rows(); // 26 (features)

        // initializing gradient trackers
        Eigen::MatrixXf dW = Eigen::MatrixXf::Zero(filter_size, filter_size);
        float db = dZ.sum(); // dbias = summation(error)
        
        // to hold error we pass backward
        Eigen::MatrixXf dX = Eigen::MatrixXf::Zero(last_input.rows(), last_input.cols());

        // Reverse sliding window to output_size
        for (int i = 0; i < out_size; ++i) {
            for (int j = 0; j < out_size; ++j) {
                // 3x3 patch from img
                Eigen::MatrixXf patch = last_input.block(i, j, filter_size, filter_size);
                
                // dW and dE computation
                dW += patch * dZ(i, j);
                dX.block(i, j, filter_size, filter_size) += filter * dZ(i, j);
            }
        }

        // weight updation
        filter -= learning_rate * dW;
        bias -= learning_rate * db;

        return dX;
    }
};


/**
 * @class MaxPool2D
 * 
 * @brief A 2D Max Pooling Layer for spatial downsampling (2x2 window, Stride 2).
 * compression feature maps, extracting max val (Eigen has a function for this) from 
 * non overlapping 2x2 patches, effectively cutting the spatial dimensions in half... 
 * nice for CPU compute, pretty much a required block here. 
 * it creates and stores binary 'mask' during fwd pass. 
 * the 'mask' stores the local co-ords so that gradient can be routed back to the 
 * 'winning' pixed during backward pass. 
 */
class MaxPool2D {
private:
    Eigen::MatrixXf mask; // 'mask' cache
    int input_rows;
    int input_cols;

public:
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) {
        input_rows = input.rows();
        input_cols = input.cols();
        int out_size = input_rows / 2; 
        
        Eigen::MatrixXf output = Eigen::MatrixXf::Zero(out_size, out_size);
        
        // reset mask to 0 for new pass
        mask = Eigen::MatrixXf::Zero(input_rows, input_cols);

        for (int i = 0; i < out_size; ++i) {
            for (int j = 0; j < out_size; ++j) {
                Eigen::MatrixXf patch = input.block(i * 2, j * 2, 2, 2);
                
                // finding max and its local location
                int max_r, max_c;
                output(i, j) = patch.maxCoeff(&max_r, &max_c); 
                
                // 'winner' (saving global co-ords)
                mask(i * 2 + max_r, j * 2 + max_c) = 1.0f;
            }
        }
        return output;
    }

    // backward pass
    Eigen::MatrixXf backward(const Eigen::MatrixXf& dZ) {
        // empty mat with 26x26 because that is the feature size that we have used
        Eigen::MatrixXf dX = Eigen::MatrixXf::Zero(input_rows, input_cols);
        int out_size = dZ.rows();

        for (int i = 0; i < out_size; ++i) {
            for (int j = 0; j < out_size; ++j) {
                // extract the 2x2 patch from our saved mask
                Eigen::MatrixXf mask_patch = mask.block(i * 2, j * 2, 2, 2);

                dX.block(i * 2, j * 2, 2, 2) = mask_patch * dZ(i, j);
            }
        }
        return dX;
    }
};


/**
 * @class ReLU2D
 * @brief _/ (smiles)
 * important part: again implements the cache logic to correctly evaluate the 
 * derivative during back propagation.
 */
class ReLU2D {
private:
    Eigen::MatrixXf last_input; // cache
public:
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) {
        last_input = input;
        return input.cwiseMax(0.0f);
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& dZ) {
        Eigen::MatrixXf dX = dZ;
        // remove grads below 0 to be 0
        for(int i = 0; i < dX.rows(); ++i) {
            for(int j = 0; j < dX.cols(); ++j) {
                if(last_input(i, j) <= 0.0f) {
                    dX(i, j) = 0.0f; 
                }
            }
        }
        return dX;
    }
};