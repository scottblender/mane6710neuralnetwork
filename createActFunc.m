% Computes the activation function output based on the specified type.
%
% Inputs:
%   w        - Weight matrix (size: [number of units in previous layer 
% x number of units in current layer])
%   x        - Input vector (size: [number of features x number of 
% examples])
%   b        - Bias vector (size: [number of units in current layer x 1])
%   func     - String specifying the activation function to use. It can be:
%             'sigmoid', 'tanh', 'ReLu', 'softmax', or 'identity'.
%
% Outputs:
%   actfunc  - Output after applying the activation function. It has the 
% same size as `z`.
%
% The function computes the weighted sum of inputs `z = w * x + b`, and 
% then applies
% the corresponding activation function based on the value of `func`.

function actfunc = createActFunc(w, x, b, func)
z = w * x + b;  % Compute weighted sum of inputs plus bias

% Apply the sigmoid function if 'sigmoid' is selected
if strcmp(func, 'sigmoid') == 1
    actfunc = 1 ./ (1 + exp(-z));
% Apply the tanh function if 'tanh' is selected
elseif strcmp(func, 'tanh') == 1
    actfunc = tanh(z);
% Apply the ReLU function if 'ReLu' is selected
elseif strcmp(func, 'ReLu') == 1
    actfunc = max(0, z);
% Apply the softmax function if 'softmax' is selected
elseif strcmp(func, 'softmax') == 1
    z = z - max(z, [], 1);  % Numerical stability trick for softmax
    actfunc = softmax(z);  % Apply softmax function
% Apply the identity function if 'identity' is selected
elseif strcmp(func, 'identity') == 1
    actfunc = z;  % No activation, return the input directly
end
end
