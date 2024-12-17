% Performs forward propagation and computes the gradients (dw, db) 
% as well as the cost for a given activation function.
%
% Inputs:
%   w        - Weight matrix (size: [number of units in previous layer 
%              x number of units in current layer])
%   b        - Bias vector (size: [number of units in current layer x 1])
%   x        - Input data matrix (size: [number of features x number 
%              of examples])
%   y        - True labels (size: [number of classes x number of examples])
%   func     - String specifying the activation function to use. It can be:
%             'sigmoid', 'tanh', 'ReLu', 'softmax', or 'identity'.
%             'identity' is used for raw logits in multiclass classification.
%
% Outputs:
%   dw       - Gradient of the loss with respect to the weights (size: 
% same as w)
%   db       - Gradient of the loss with respect to the bias (size: same as b)
%   cost     - Scalar value representing the cost (loss) for the current 
% iteration.
%
% The function computes the activation `z_hat = w * x + b`, then computes 
% the gradients and cost based on the chosen activation function (including 
% identity), using the cross-entropy loss for multiclass classification.

function [dw, db, cost] = createPropagation(w, b, x, y, func)
    % Compute activation using the specified function
    z_hat = createActFunc(w, x, b, func);  % Compute activation 
    % (identity or others)

    % --- Handle different activation functions ---
    if strcmp(func, 'softmax') == 1
        dA = z_hat - y;  % Softmax + Cross-Entropy error term
    elseif strcmp(func, 'sigmoid') == 1
        dA = z_hat - y;  % Sigmoid + Cross-Entropy error term
    elseif strcmp(func, 'ReLu') == 1
        dA = (z_hat > 0) .* (z_hat - y);  % ReLU derivative for error term
    elseif strcmp(func, 'tanh') == 1
        dA = (1 - z_hat.^2) .* (z_hat - y);  % Tanh derivative for error term
    elseif strcmp(func, 'identity') == 1
        dA = z_hat - y;  % Identity function: no non-linearity, same as 
        % error term
    end

    % Number of training examples
    m = size(x, 2);  
    
    % Compute gradients
    dw = (1 / m) * (dA * x.');  % Gradient of the loss with respect to 
    % the weights
    db = mean(dA, 2);  % Gradient of the loss with respect to the bias

    % --- Compute cost based on the activation function ---
    if strcmp(func, 'sigmoid') == 1
        cost = -mean(sum(y .* log(z_hat + eps) + (1 - y) .* ...
            log(1 - z_hat + eps), 1));  % Sigmoid cross-entropy cost
    elseif strcmp(func, 'tanh') == 1
        cost = -mean(sum(y .* log(z_hat + eps), 1));  % Tanh cross-entropy cost
    elseif strcmp(func, 'ReLu') == 1
        cost = -mean(sum(y .* log(z_hat + eps), 1));  % ReLU cross-entropy cost
    elseif strcmp(func, 'softmax') == 1
        cost = -mean(sum(y .* log(z_hat + eps), 1));  % Softmax 
        % cross-entropy cost
    elseif strcmp(func, 'identity') == 1
        cost = -mean(sum(y .* log(z_hat + eps), 1));  % Cross-entropy 
        % for identity function
    end
end
