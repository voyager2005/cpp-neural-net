#include <iostream>
#include <vector>
#include <Eigen/Dense>

// the same dense layer from before
class DenseLayer {
private:
    Eigen::MatrixXf weights;
    Eigen::VectorXf biases;
    Eigen::VectorXf last_input; // required input for the back propagation

public:
    DenseLayer(int input_size, int output_size) {
        weights = Eigen::MatrixXf::Random(output_size, input_size);
        biases = Eigen::VectorXf::Zero(output_size);
    }

    Eigen::VectorXf forward(const Eigen::VectorXf& inputs) {
        last_input = inputs; // saving previous inptus for back propagation 
        return (weights * inputs) + biases;
    }

    // gradient of sigmoid
    Eigen::VectorXf backward(const Eigen::VectorXf& dZ, float learning_rate) {
        // G for w and b
        Eigen::MatrixXf dW = dZ * last_input.transpose();
        Eigen::VectorXf dB = dZ;
        
        // error computation
        Eigen::VectorXf dX = weights.transpose() * dZ;

        // on line learning weight updation
        weights -= learning_rate * dW;
        biases  -= learning_rate * dB;

        return dX;
    }
};

// same ReLU activation from before
class ReLU {
private:
    Eigen::VectorXf last_input; // cahce
public:
    Eigen::VectorXf forward(const Eigen::VectorXf& inputs) {
        last_input = inputs;
        return inputs.cwiseMax(0.0f);
    }

    Eigen::VectorXf backward(const Eigen::VectorXf& dOut) {
        Eigen::VectorXf dZ = dOut;
        // The derivative of ReLU is 1 if input > 0, else 0. (_/ ts)
        for(int i = 0; i < dZ.size(); ++i) {
            if(last_input(i) <= 0) {
                dZ(i) = 0.0f; 
            }
        }
        return dZ;
    }
};

// MSE loss 
class MSELoss {
public:
    float forward(const Eigen::VectorXf& pred, const Eigen::VectorXf& target) {
        return (pred - target).squaredNorm() / pred.size();
    }

    // The derivative of Mean Squared Error
    Eigen::VectorXf backward(const Eigen::VectorXf& pred, const Eigen::VectorXf& target) {
        return 2.0f * (pred - target) / pred.size();
    }
};


int main() {
    std::cout << "Training MLP on XOR data\n\n";

    // 1. The XOR Dataset
    std::vector<Eigen::VectorXf> X_train = {
        (Eigen::VectorXf(2) << 0, 0).finished(),
        (Eigen::VectorXf(2) << 0, 1).finished(),
        (Eigen::VectorXf(2) << 1, 0).finished(),
        (Eigen::VectorXf(2) << 1, 1).finished()
    };
    
    std::vector<Eigen::VectorXf> Y_train = {
        (Eigen::VectorXf(1) << 0).finished(),
        (Eigen::VectorXf(1) << 1).finished(),
        (Eigen::VectorXf(1) << 1).finished(),
        (Eigen::VectorXf(1) << 0).finished()
    };

    // 2. 2 Inputs -> 4 Hidden Nodes -> 1 Output Node
    DenseLayer hidden_layer(2, 4);
    ReLU relu;
    DenseLayer output_layer(4, 1);
    MSELoss criterion;

    // 3. Hyperparameters
    float learning_rate = 0.05f;
    int epochs = 100; // crazy overkill tbh, i think it will converge before 10th epoch

    // 4. training loop
    for (int epoch = 0; epoch <= epochs; ++epoch) {
        float epoch_loss = 0.0f;

        for (size_t i = 0; i < X_train.size(); ++i) {
            // fwd pass
            Eigen::VectorXf z1 = hidden_layer.forward(X_train[i]);
            Eigen::VectorXf a1 = relu.forward(z1);
            Eigen::VectorXf pred = output_layer.forward(a1); // Z2

            epoch_loss += criterion.forward(pred, Y_train[i]);

            // back prop
            Eigen::VectorXf dLoss = criterion.backward(pred, Y_train[i]);
            Eigen::VectorXf dA1   = output_layer.backward(dLoss, learning_rate); // Back through output layer
            Eigen::VectorXf dZ1   = relu.backward(dA1);                          // Back through ReLU
            hidden_layer.backward(dZ1, learning_rate);                           // Back through hidden layer
        }

        // 4th epoch print
        if (epoch % 4 == 0) {
            std::cout << "Epoch " << epoch << " | Average Loss: " << (epoch_loss / 4.0f) << "\n";
        }
    }

    // 5. testing
    std::cout << "\nCurassifcation Reporto!\n";
    for (size_t i = 0; i < X_train.size(); ++i) {
        Eigen::VectorXf z1 = hidden_layer.forward(X_train[i]);
        Eigen::VectorXf a1 = relu.forward(z1);
        Eigen::VectorXf pred = output_layer.forward(a1);
        
        std::cout << "Input: [" << X_train[i].transpose() << "] -> "
                  << "Prediction: " << pred(0) 
                  << " (Target: " << Y_train[i](0) << ")\n";
    }

    return 0;
}