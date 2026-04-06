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

class ReLU {
public:
    // replaces all negs with 0, ReLU activation function _/
    Eigen::VectorXf forward(const Eigen::VectorXf& inputs) {
        return inputs.cwiseMax(0.0f);
    }
};

class MSELoss {
public:
    // Calculates the Mean Squared Error between predictions and targets
    float forward(const Eigen::VectorXf& predictions, const Eigen::VectorXf& targets) {
        Eigen::VectorXf difference = predictions - targets;
        return difference.squaredNorm() / difference.size();
    }
};

int main(){
    // display statement to test neural network 
    std::cout << "Forward Pass Testing with ReLU Activation :)\n";

    // declaration and initialization
    DenseLayer layer1(3, 2);
    ReLU relu1;
    MSELoss criterion;

    // creating dummy input data
    Eigen::VectorXf inputs(3); // 3 is basically the number of features
    inputs << 1.0f, -2.0f, 3.0f;

    // defining dummy output for this given set of input 
    Eigen::VectorXf target(2);
    target << 0.0f, 1.0f;

    // passing the data through the layers
    Eigen::VectorXf z = layer1.forward(inputs);
    Eigen::VectorXf a = relu1.forward(z);

    //computing loss
    float loss = criterion.forward(a, target);

    // displaying the results
    std::cout << "1. Input Data:         [" << inputs.transpose() << "]\n";
    std::cout << "2. Target Output:      [" << target.transpose() << "]\n";
    std::cout << "3. Network Prediction: [" << a.transpose() << "]\n";
    std::cout << "------------------------------------------\n";
    std::cout << ">> Final Network Loss: " << loss << "\n\n";

    return 0;
}