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
%   func1    - String specifying the activation function for the first layer 
%              ('sigmoid', 'tanh', 'ReLu', 'identity')
%   func2    - String specifying the activation function for the second layer 
%              ('sigmoid', 'tanh', 'ReLu', 'identity')
%   sample_size - The number of random samples to use for gradient checking 
%                 (to check numerical gradients for the first layer).
%
% Outputs:
%   dw1_cs   - Complex-step gradient of the loss with respect to the weights 
%              of the first layer (size: same as w1)
%   db1_cs   - Complex-step gradient of the loss with respect to the bias 
%              of the first layer (size: same as b1)
%   dw2_cs   - Complex-step gradient of the loss with respect to the weights 
%              of the second layer (size: same as w2)
%   db2_cs   - Complex-step gradient of the loss with respect to the bias 
%              of the second layer (size: same as b2)
%
% This function computes the forward pass for a neural network, then checks 
% the gradients of the loss with respect to the weights and biases using 
% both analytical gradients and complex-step numerical differentiation. 
% It also compares the results to ensure correctness.
function [dw1_cd, db1_cd, dw2_cs, db2_cs] = checkGradients(w1, b1, w2, b2, ...
    x, y, func1, func2, sample_size)
h_cs = 1e-30; % Adjusted small imaginary step for complex-step differentiation
h_cd = 1e-5;

% --- Forward Pass ---
a1 = createActFunc(w1, x, b1, func1); % Activations for the hidden layer
a2 = createActFunc(w2, a1, b2, func2); % Activations for the output layer

% --- Analytical Gradients ---
[dw2, db2, ~] = createPropagation(w2, b2, a1, y, func2); % Compute 
% gradients for the output layer
dA2 = a2 - y; % Error in the second layer (output layer)

% Compute dA1 (the gradient for the hidden layer)
if strcmp(func1, 'sigmoid')
    dA1 = (w2' * dA2) .* a1 .* (1 - a1); % Sigmoid derivative: a1 * (1 - a1)
elseif strcmp(func1, 'tanh')
    dA1 = (w2' * dA2) .* (1 - a1.^2); % Tanh derivative: (1 - a1^2)
elseif strcmp(func1, 'ReLu')
    dA1 = (w2' * dA2) .* (a1 > 0); % ReLU derivative: a1 > 0
elseif strcmp(func1, 'identity')
    dA1 = w2' * dA2; % Identity derivative is 1, so no change
end

m = size(x, 2); % Number of examples
dw1 = (1 / m) * (dA1 * x'); % Gradient of the loss with respect to the 
% weights of the first layer
db1 = mean(dA1, 2); % Gradient of the loss with respect to the bias of 
% the first layer

% --- Numerical Gradients for Output Layer ---
dw2_cs = zeros(size(w2));
db2_cs = zeros(size(b2));

for i = 1:numel(w2)
    w_cs = w2; % Copy weights
    w_cs(i) = w_cs(i) + complex(0.0, h_cs); % Add small imaginary step
    [~, ~, cost] = createPropagation(w_cs, b2, a1, y, func2); % Recompute cost
    dw2_cs(i) = imag(cost) / h_cs; % Gradient from complex step
end

for j = 1:numel(b2)
    b_cs = b2; % Copy biases
    b_cs(j) = b_cs(j) + complex(0.0, h_cs); % Add small imaginary step
    [~, ~, cost] = createPropagation(w2, b_cs, a1, y, func2); % Recompute cost
    db2_cs(j) = imag(cost) / h_cs; % Gradient from complex step
end

% --- Numerical Gradients for Hidden Layer (with sampling) ---
num_weights_w1 = numel(w1);
num_biases_b1 = numel(b1);
sampled_indices_w1 = randperm(num_weights_w1, min(sample_size, ...
    num_weights_w1)); % Randomly sample weights
sampled_indices_b1 = randperm(num_biases_b1, min(sample_size, ...
    num_biases_b1)); % Randomly sample biases

dw1_cd = zeros(size(w1));
db1_cd = zeros(size(b1));

% Compute numerical gradients for sampled weights of w1 using central difference
for idx = sampled_indices_w1
    w_plus = w1; % Copy original weights
    w_minus = w1; % Copy original weights
    
    % Perturb in both positive and negative directions
    w_plus(idx) = w_plus(idx) + h_cd; 
    w_minus(idx) = w_minus(idx) - h_cd;
    
    % Compute activations for perturbed weights (w+ and w-)
    a1_plus = createActFunc(w_plus, x, b1, func1); % Activations for w_plus
    a2_plus = createActFunc(w2, a1_plus, b2, func2); % Activations for output 
    % layer
    [~, ~, cost_plus] = createPropagation(w2, b2, a1_plus, y, func2); % Cost 
    % for w_plus
    
    a1_minus = createActFunc(w_minus, x, b1, func1); % Activations for w_minus
    a2_minus = createActFunc(w2, a1_minus, b2, func2); % Activations for 
    % output layer
    [~, ~, cost_minus] = createPropagation(w2, b2, a1_minus, ...
        y, func2); % Cost for w_minus
    
    % Central difference gradient for this weight
    dw1_cd(idx) = (cost_plus - cost_minus) / (2 * h_cd); 

    % Debugging prints
    disp(['Perturbed Weight (w' num2str(idx) '): ', num2str(w1(idx))]);
    disp(['Original Cost (w' num2str(idx) '): ', num2str(cost_plus), ...
        ' | ', num2str(cost_minus)]);
    disp(['Central Difference Gradient (dw1_cs' num2str(idx) '): ', ...
        num2str(dw1_cd(idx))]);
    disp(['Regular Gradient (dw1' num2str(idx) '): ', ...
        num2str(dw1(idx))]); % Print regular gradient
end

% Compute numerical gradients for sampled biases of b1 using central difference
for idx = sampled_indices_b1
    b_plus = b1; % Copy original biases
    b_minus = b1; % Copy original biases
    
    % Perturb in both positive and negative directions
    b_plus(idx) = b_plus(idx) + h_cd; 
    b_minus(idx) = b_minus(idx) - h_cd;
    
    % Compute activations for perturbed biases (b+ and b-)
    a1_plus = createActFunc(w1, x, b_plus, func1); % Activations for b_plus
    a2_plus = createActFunc(w2, a1_plus, b2, func2); % Activations for output 
    % layer
    [~, ~, cost_plus] = createPropagation(w2, b2, a1_plus, y, func2); % Cost 
    % for b_plus
    
    a1_minus = createActFunc(w1, x, b_minus, func1); % Activations for b_minus
    a2_minus = createActFunc(w2, a1_minus, b2, func2); % Activations for 
    % output layer
    [~, ~, cost_minus] = createPropagation(w2, b2, a1_minus, y, func2); % Cost 
    % for b_minus
    
    % Central difference gradient for this bias
    db1_cd(idx) = (cost_plus - cost_minus) / (2 * h_cd);

    % Debugging prints
    disp(['Perturbed Bias (b' num2str(idx) '): ', num2str(b1(idx))]);
    disp(['Original Cost (b' num2str(idx) '): ', num2str(cost_plus), ...
        ' | ', num2str(cost_minus)]);
    disp(['Central Difference Gradient (db1_cs' num2str(idx) '):' ...
        ' ', num2str(db1_cd(idx))]);
    disp(['Regular Gradient (db1' num2str(idx) '): ', ...
        num2str(db1(idx))]); % Print regular gradient
end


% Display max and mean differences between the analytical and complex-step 
% gradients
disp('Output Layer Gradients:');
disp(['Max Difference (Weights): ', num2str(max(abs(dw2(:) - dw2_cs(:))))]);
disp(['Max Difference (Biases): ', num2str(max(abs(db2(:) - db2_cs(:))))]);
disp(['Mean Difference (Weights): ', num2str(mean(abs(dw2(:) - dw2_cs(:))))]);
disp(['Mean Difference (Biases): ', num2str(mean(abs(db2(:) - db2_cs(:))))]);

% Display max and mean differences for hidden layer gradients (sampled)
disp('Hidden Layer Gradients (Sampled):');
disp(['Max Difference (Weights): ', num2str(max(abs(dw1(sampled_indices_w1) ...
    - dw1_cd(sampled_indices_w1))))]);
disp(['Max Difference (Biases): ', num2str(max(abs(db1(sampled_indices_b1) ...
    - db1_cd(sampled_indices_b1))))]);
disp(['Mean Difference (Weights): ', num2str(mean(abs(dw1(sampled_indices_w1) ...
    - dw1_cd(sampled_indices_w1))))]);
disp(['Mean Difference (Biases): ', num2str(mean(abs(db1(sampled_indices_b1) ...
    - db1_cd(sampled_indices_b1))))]);
end