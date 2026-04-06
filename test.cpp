#include <iostream>
#include <Eigen/Dense> // This includes the core Eigen matrix math

int main() {
    std::cout << "=== Eigen Diagnostic ===" << std::endl;
    
    // Create a 2x3 matrix of floats and initialize it with random weights
    Eigen::MatrixXf weights = Eigen::MatrixXf::Random(2, 3);
    
    // Create a 3x1 vector (like an input feature vector)
    Eigen::VectorXf inputs(3);
    inputs << 1.0f, 2.0f, 3.0f;
    
    // Perform a matrix-vector multiplication (the core of a neural network forward pass!)
    Eigen::VectorXf output = weights * inputs;
    
    std::cout << "Weights Matrix:\n" << weights << "\n\n";
    std::cout << "Input Vector:\n" << inputs << "\n\n";
    std::cout << "Output (Weights * Inputs):\n" << output << std::endl;
    std::cout << "\nEigen is fully configured and ready!" << std::endl;
    
    return 0;
}