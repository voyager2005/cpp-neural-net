/**
 * @file main.cpp
 * @brief Convolutional Neural Network (CNN) from scratch in cpp using Eigen(link: https://eigen.tuxfamily.org/).
 * here we load the main MNIST dataset in the form of raw binary files, which were taken from kaggle, and construct a CNN pipeline
 * (Conv2D -> ReLU -> MaxPool2D -> Dense -> Softmax), and trains the network using 
 * Stochastic Gradietn Descent with online learning.
 * 
 * update after training: 
 * the model achieved an accuracy of: 
    Step 60000/60000 | Avg Loss: 0.246919 | Accuracy: 93.8%
    Training Epoch Complete.
 * 
 * Note to self: read the to author section in mnist_loader.hpp (that has more work to do)
 */

#include <iostream>
#include "mnist_loader.hpp"
#include "cnn_layer.hpp"

/**
 * @class DenseLayer
 * @brief A fully connected (dense) neural network layer
 * * Connects every input node to every output node using a weight matrix and bias vector
 */
class DenseLayer {
private:
    Eigen::MatrixXf weights;
    Eigen::VectorXf biases;
    Eigen::VectorXf last_input; 

public:
    /**
     * @brief Constructor for the DenseLayer
     * @param input_size number of input features (1D vector size)
     * @param output_size number of output neurons
     */
    DenseLayer(int input_size, int output_size) {
        weights = Eigen::MatrixXf::Random(output_size, input_size) * 0.1f; 
        biases = Eigen::VectorXf::Zero(output_size);
    }
    
    /**
     * @brief computes the forward pass (z = w*x + b)
     * @param inputs a 1D column vector of input features
     * @return a 1D column vector of the layer's output
     */
    Eigen::VectorXf forward(const Eigen::VectorXf& inputs) {
        last_input = inputs; 
        return (weights * inputs) + biases;
    }

    /**
     * @brief computes the backward pass, calculates gradients, and updates weights
     * @param dZ the error gradient flowing backward from the subsequent layer
     * @param learning_rate the step size for the gradient descent update
     * @return the error gradient (dX) to pass backward to the preceding layer
     */
    Eigen::VectorXf backward(const Eigen::VectorXf& dZ, float learning_rate) {
        Eigen::MatrixXf dW = dZ * last_input.transpose();
        Eigen::VectorXf dB = dZ;
        Eigen::VectorXf dX = weights.transpose() * dZ;

        weights -= learning_rate * dW;
        biases  -= learning_rate * dB;

        return dX;
    }
};

/**
 * @class Softmax
 * @brief applies Softmax activation and computes Categorical Cross-Entropy loss
 */
class Softmax {
private:
    Eigen::VectorXf last_probs;

public:
    /**
     * @brief computes the probability distribution across all classes
     * @param input raw logit predictions from the preceding Dense layer
     * @return a 1D vector of probabilities summing to 1.0.
     */
    Eigen::VectorXf forward(const Eigen::VectorXf& input) {
        Eigen::VectorXf exp_vals = (input.array() - input.maxCoeff()).exp();
        last_probs = exp_vals / exp_vals.sum();
        return last_probs;
    }

    /**
     * @brief computes the derivative of Cross-Entropy Loss with Softmax
     * @param target_label the integer index of the correct ground-truth class
     * @return a 1D vector representing the initial error gradient (Predictions - Targets)
     */
    Eigen::VectorXf backward(int target_label) {
        Eigen::VectorXf dZ = last_probs;
        dZ(target_label) -= 1.0f; 
        return dZ;
    }

    /**
     * @brief calculates the scalar loss value for monitoring network performance
     * @param target_label the integer index of the correct ground-truth class
     * @return the Categorical Cross-Entropy loss as a float
     */
    float calculate_loss(int target_label) {
        return -std::log(last_probs(target_label) + 1e-7); 
    }
};

/**
 * @brief main execution function
 * handles data loading, architecture initialization, and the training loop
 */
int main() {
    std::cout << "Binary MNIST Extraction\n\n";

    std::string train_images_path = R"(C:\Users\Akshat - Personal\Visual Studio Code\cpp-neural-net\data\train-images.idx3-ubyte)";
    std::string train_labels_path = R"(C:\Users\Akshat - Personal\Visual Studio Code\cpp-neural-net\data\train-labels.idx1-ubyte)";
    
    std::cout << "Loading images into memory...\n";
    std::vector<Eigen::VectorXf> train_images = read_mnist_images(train_images_path);
    std::vector<int> train_labels = read_mnist_labels(train_labels_path);

    std::cout << "\nImages Loaded: " << train_images.size();
    std::cout << "\nLabels Loaded: " << train_labels.size() << "\n\n";

    if (train_images.size() > 0) {
        std::cout << "--- First Image in Dataset (Label: " << train_labels[0] << ")\n\n";
        Eigen::VectorXf first_image = train_images[0];
        
        for (int row = 0; row < 28; ++row) {
            for (int col = 0; col < 28; ++col) {
                float pixel_value = first_image(row * 28 + col);
                if (pixel_value > 0.3f) {
                    std::cout << "##"; 
                } else {
                    std::cout << "..";
                }
            }
            std::cout << "\n";
        }
    }

    Conv2D conv_layer; 
    ReLU2D relu_layer;
    MaxPool2D pool_layer;
    DenseLayer final_layer(169, 10); 
    Softmax softmax_layer;

    int total_dataset_size = train_images.size();
    std::cout << "\n=== Beginning Training (" << total_dataset_size << " Images) ===\n";
    
    float learning_rate = 0.01f;
    float total_loss = 0.0f;
    int correct_predictions = 0;

    for (int step = 0; step < total_dataset_size; ++step) {
        Eigen::Map<Eigen::MatrixXf> image_2d_mapped(train_images[step].data(), 28, 28);
        Eigen::MatrixXf input_image = image_2d_mapped.transpose();
        int target_label = train_labels[step];

        Eigen::MatrixXf f_map       = conv_layer.forward(input_image);
        Eigen::MatrixXf a_map       = relu_layer.forward(f_map);
        Eigen::MatrixXf p_map       = pool_layer.forward(a_map);
        
        Eigen::Map<Eigen::VectorXf> flat_vec(p_map.data(), p_map.size());
        
        Eigen::VectorXf raw_preds   = final_layer.forward(flat_vec);
        Eigen::VectorXf probs       = softmax_layer.forward(raw_preds);

        total_loss += softmax_layer.calculate_loss(target_label);
        
        int predicted_label;
        probs.maxCoeff(&predicted_label); 
        if (predicted_label == target_label) correct_predictions++;

        Eigen::VectorXf dZ_out = softmax_layer.backward(target_label);
        Eigen::VectorXf dFlat = final_layer.backward(dZ_out, learning_rate);

        Eigen::Map<Eigen::MatrixXf> dPool_mapped(dFlat.data(), 13, 13);
        Eigen::MatrixXf dPool = dPool_mapped; 

        Eigen::MatrixXf dRelu = pool_layer.backward(dPool);
        Eigen::MatrixXf dConv = relu_layer.backward(dRelu);
        conv_layer.backward(dConv, learning_rate);

        if ((step + 1) % 1000 == 0) {
            std::cout << "Step " << step + 1 << "/" << total_dataset_size << " | "
                      << "Avg Loss: " << (total_loss / 1000.0f) << " | "
                      << "Accuracy: " << (correct_predictions / 1000.0f) * 100.0f << "%\n";
            
            total_loss = 0.0f;
            correct_predictions = 0;
        }
    }
    std::cout << "\nTraining Epoch Complete.\n";

    return 0;
}