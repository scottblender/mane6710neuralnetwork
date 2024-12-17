% Set default font size and line width for plots
set(0, 'DefaultAxesFontSize', 16);
set(0, 'DefaultLineLineWidth', 1.5);

% Clear workspace, close figures, and clear command window
clear
close all
clc

% Load MNIST dataset
[trainingdata, traininglabels, testingdata, testinglabels] = readMNIST();

% Normalize pixel values to range [0, 1]
trainingdata = double(trainingdata) / 255;
testingdata = double(testingdata) / 255;

% Flatten the 28x28 images into 784x1 column vectors
trainingdata = reshape(trainingdata, [], size(trainingdata, 3));
testingdata = reshape(testingdata, [], size(testingdata, 3));

% Convert labels to one-hot encoding
traininglabels = full(ind2vec(traininglabels' + 1));
testinglabels = full(ind2vec(testinglabels' + 1));

% Set network parameters
input_size = 784;     % Number of input features (28x28 images)
hidden_size = 128;    % Number of hidden units
output_size = 10;     % Number of output classes (digits 0-9)

% Initialize weights and biases using He initialization
w1 = randn(hidden_size, input_size) * sqrt(2 / input_size);
b1 = zeros(hidden_size, 1);
w2 = randn(output_size, hidden_size) * sqrt(2 / hidden_size);
b2 = zeros(output_size, 1);

% Set training parameters
num_iterations = 500;  % Number of iterations (epochs)
learning_rate = 0.1;   % Learning rate
func1 = 'ReLu';        % Activation function for hidden layer
func2 = 'softmax';     % Activation function for output layer

% Train the network by updating weights and biases
[w1, b1, dw1, db1, w2, b2, dw2, db2, costs] = createNetworkUpdates(w1, ...
    b1, w2, b2, trainingdata, traininglabels, num_iterations, ...
    learning_rate, func1, func2);

% Perform gradient checking
fprintf('Performing Gradient Check...\n');
[dw1_cs, db1_cs, dw2_cs, db2_cs] = checkGradients(w1, b1, w2, b2, ...
    trainingdata(:, 1:100), traininglabels(:, 1:100), func1, func2, 2000);
fprintf('Gradient Check Completed!\n');

% Forward propagation for the test data
a1_test = createActFunc(w1, testingdata, b1, func1);
a2_test = createActFunc(w2, a1_test, b2, func2);

% Get predicted class labels by selecting the max value from softmax output
[~, predicted_labels] = max(a2_test, [], 1);

% Convert predictions from one-hot encoding back to class labels (0-9)
predicted_labels = predicted_labels - 1;

% Convert true labels from one-hot encoding back to class labels (0-9)
[~, true_labels] = max(testinglabels, [], 1);
true_labels = true_labels - 1;

% Calculate and print test accuracy
accuracy = mean(predicted_labels == true_labels);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% Plot loss function during training
figure;
plot(1:num_iterations, costs);
xlabel('Iteration');
ylabel('Loss');
title('Loss Function during Training');

% Visualize some test set predictions
figure
for i = 1:10
    idx = randi(size(testingdata, 2));  % Random index for test data
    image = reshape(testingdata(:, idx), [28, 28]);  % Reshape back 
    % to 28x28 image
    predicted_label = predicted_labels(idx);  % Get predicted label
    true_label = true_labels(idx);  % Get true label
    subplot(2, 5, i);
    imshow(image);
    title(sprintf('True: %d, Pred: %d', true_label, predicted_label));  
    % Display image with labels
end
