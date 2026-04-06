#include <iostream>
#include <Eigen/Dense>

class DenseLayer {
private: 
    Eigen::MatrixXf weights;
    Eigen::VectorXf biases;

public: 

    // constructor that sets up the layer dimensions and initializes values 
    DenseLayer(int input_size, int output_size){
        // declaration and initialization 
        weights = Eigen::MatrixXf::Random(output_size, input_size);
        biases = Eigen::VectorXf::Zero(output_size);
    }

    // forward pass z = wx + b (perceptron back again :( )
    Eigen::VectorXf forward(const Eigen::VectorXf& inputs){
        return (weights * inputs) + biases;
    }

    // adding an intermediate helper function to just peek into the layer's parameters
    void print_info(){
        std::cout << "Weights:\n" << weights << "\n\n";
        std::cout << "Biases:\n" << biases << "\n\n";
    }
};

int main(){
    // display statement to test neural network 
    std::cout << "Neural Network Testing :)\n";

    // declaration and initializing inputs
    DenseLayer layer1(3, 2);

    // checking layer weights and biases
    std::cout << "Layer Initialization";
    layer1.print_info();

    // creating dummy input data
    Eigen::VectorXf inputs(3); // 3 is basically the number of features
    inputs << 1.0f, 2.0f, 3.0f;

    // passing the data through the layers
    Eigen::VectorXf output = layer1.forward(inputs);

    // displaying the results
    std::cout <<"Forward Pass Results:\n";
    std::cout << "Input Data:\n" << inputs << "\n\n";
    std::cout << "Layer Output:\n" << output << std::endl;

    return 0;
}