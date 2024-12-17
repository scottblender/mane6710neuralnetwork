% Reads the MNIST dataset, which contains image and label data for both
% training and testing. The function extracts and loads image and label 
% files for training and testing, and returns them as separate arrays.
%
% Inputs: 
%   None. The function loads the image and label files directly.
%
% Outputs:
%   trainingimages  - 3D array of training images, with dimensions: 
% [rows x cols x number of training images]
%   traininglabels  - 1D array of labels for the training images
%   testingimages   - 3D array of testing images, with dimensions: 
% [rows x cols x number of testing images]
%   testinglabels   - 1D array of labels for the testing images
%
% The function reads the MNIST dataset from four separate files (train images, 
% train labels, test images, and test labels), processes the data, and returns 
% the image data and corresponding labels for both the training and testing sets.
%--------------------------------------------------------------------------
function [trainingimages, traininglabels, testingimages, testinglabels] = ...
    readMNIST()
% Open the training and testing image and label files
id1 = fopen("train-images-idx3-ubyte/train-images.idx3-ubyte");
id2 = fopen("train-labels-idx1-ubyte/train-labels.idx1-ubyte");
id3 = fopen("t10k-images-idx3-ubyte/t10k-images.idx3-ubyte");
id4 = fopen("t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte");

% Read the magic number for the training image file and display it
A = fread(id1, 1, 'uint32');
magic1 = swapbytes(uint32(A));
fprintf('Magic Number - Images: %d\n', magic1);

% Read the magic number for the training labels file and display it
A = fread(id2, 1, 'uint32');
magic2 = swapbytes(uint32(A));
fprintf('Magic Number - Images: %d\n', magic2);

% Read the magic number for the testing image file and display it
A = fread(id3, 1, 'uint32');
magic3 = swapbytes(uint32(A));
fprintf('Magic Number - Images: %d\n', magic3);

% Read the magic number for the testing labels file and display it
A = fread(id4, 1, 'uint32');
magic4 = swapbytes(uint32(A));
fprintf('Magic Number - Images: %d\n', magic4);

% Read the total number of training images and check consistency with 
% labels file
A = fread(id1, 1, 'uint32');
trainingtotalimages = swapbytes(uint32(A));
A = fread(id2, 1, 'uint32');
if trainingtotalimages ~= swapbytes(uint32(A))
    error(['Total number of images read from images and labels files are' ...
        ' not the same']);
end

% Read the total number of testing images and check consistency with 
% labels file
A = fread(id3, 1, 'uint32');
testingtotalimages = swapbytes(uint32(A));
A = fread(id4, 1, 'uint32');
if testingtotalimages ~= swapbytes(uint32(A))
    error(['Total number of images read from images and labels files are ' ...
        'not the same']);
end

% Read the number of rows (height) of each image
A = fread(id1, 1, 'uint32');
numrows = swapbytes(uint32(A));

% Read the number of columns (width) of each image
A = fread(id1, 1, 'uint32');
numcols = swapbytes(uint32(A));

% Initialize a 3D array for storing the training images
trainingimages = zeros(numrows, numcols, trainingtotalimages, 'uint8');

% Read the pixel data for each training image and store it in the array
for k = 1 : trainingtotalimages
    A = fread(id1, numrows*numcols, 'uint8');
    trainingimages(:,:,k) = reshape(uint8(A), numcols, numrows).';
end

% Read the labels for the training images
traininglabels = fread(id2, trainingtotalimages, 'uint8');

% Close the training files
fclose(id1);
fclose(id2);

% Read the number of rows (height) of each testing image
A = fread(id3, 1, 'uint32');
numrows = swapbytes(uint32(A));

% Read the number of columns (width) of each testing image
A = fread(id3, 1, 'uint32');
numcols = swapbytes(uint32(A));

% Initialize a 3D array for storing the testing images
testingimages = zeros(numrows, numcols, testingtotalimages, 'uint8');

% Read the pixel data for each testing image and store it in the array
for k = 1 : testingtotalimages
    A = fread(id3, numrows*numcols, 'uint8');
    testingimages(:,:,k) = reshape(uint8(A), numcols, numrows).';
end

% Read the labels for the testing images
testinglabels = fread(id4, testingtotalimages, 'uint8');

% Close the testing files
fclose(id3);
fclose(id4);
end
