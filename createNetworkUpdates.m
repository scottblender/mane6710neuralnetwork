% Performs forward propagation, computes gradients, and updates weights 
% and biases 
% for a two-layer neural network using gradient descent. It stores the 
% cost for each iteration.
%
% Inputs:
%   w1       - Weight matrix for the first layer (size: [hidden_units x 
% input_features])
%   b1       - Bias vector for the first layer (size: [hidden_units x 1])
%   w2       - Weight matrix for the second layer (size: [output_classes 
% x hidden_units])
%   b2       - Bias vector for the second layer (size: [output_classes x 1])
%   x        - Input data matrix (size: [input_features x num_samples])
%   y        - True labels (size: [output_classes x num_samples])
%   num_iterations - Number of iterations (epochs) for training
%   learning_rate  - Learning rate for gradient descent
%   func1    - Activation function for the first layer (options: 'sigmoid'
% , 'tanh', 'ReLu', 'identity')
%   func2    - Activation function for the second layer (options: 'sigmoid'
% , 'tanh', 'ReLu', 'softmax', 'identity')
%
% Outputs:
%   w1       - Updated weight matrix for the first layer
%   b1       - Updated bias vector for the first layer
%   dw1      - Gradient of the loss with respect to the weights of the 
% first layer
%   db1      - Gradient of the loss with respect to the bias of the 
% first layer
%   w2       - Updated weight matrix for the second layer
%   b2       - Updated bias vector for the second layer
%   dw2      - Gradient of the loss with respect to the weights of the 
% second layer
%   db2      - Gradient of the loss with respect to the bias of the 
% second layer
%   costs    - Array containing the cost at each iteration

function [w1, b1, dw1, db1, w2, b2, dw2, db2, costs] = ...
createNetworkUpdates(w1, b1, w2, b2, x, y, num_iterations, learning_rate, ...
func1, func2)
costs = zeros(num_iterations, 1);  % Initialize costs array

for i = 1:num_iterations
    % Forward propagation for the first layer (hidden layer)
    a1 = createActFunc(w1, x, b1, func1);  % Activation for the first layer
    a2 = createActFunc(w2, a1, b2, func2);  % Activation for the 
    % second layer (output)

    % Compute gradients and cost for the output layer
    [dw2, db2, cost2] = createPropagation(w2, b2, a1, y, func2);  
    costs(i) = cost2;  % Store the cost for this iteration

    % Update weights and biases for the second layer
    w2 = w2 - learning_rate * dw2;  
    b2 = b2 - learning_rate * db2;  

    % Compute error for the output layer (cross-entropy loss)
    dA2 = a2 - y;  % Error term for output layer
    
    % For the identity function, no derivative needed in the first layer
    % Simply propagate the error backward without any modification
    dA1 = w2' * dA2;  % Backpropagate error to the first layer

    % No non-linearity for identity activation: the gradient is directly propagated
    if strcmp(func1, 'identity')
        % Identity derivative is simply 1, so no modification needed
        dA1 = dA1;  % Just keep the propagated error
    elseif strcmp(func1, 'sigmoid')
        dA1 = (w2' * dA2) .* a1 .* (1 - a1);  % Compute error for hidden 
        % layer with sigmoid
    elseif strcmp(func1, 'tanh')
        dA1 = (w2' * dA2) .* (1 - a1.^2);  % Compute error for hidden 
        % layer with tanh
    elseif strcmp(func1, 'ReLu')
        dA1 = (w2' * dA2) .* (a1 > 0);  % Compute error for hidden 
        % layer with ReLU
    end

    % Compute gradients for the first layer
    m = size(x, 2);  % Number of training samples
    dw1 = (1 / m) * (dA1 * x');  % Gradient for weights of the first layer
    db1 = mean(dA1, 2);  % Gradient for biases of the first layer

    % Update weights and biases for the first layer
    w1 = w1 - learning_rate * dw1;  
    b1 = b1 - learning_rate * db1;  

    % Print cost every 100 iterations
    if mod(i, 100) == 0
        fprintf('Iteration %d: Cost = %.4f\n', i, cost2);
    end
end
end
