function [J grad] = nnCostFunction(nn, ... %nn_params
                                   il, ... %input_layer_size
                                   hl, ... %hidden_layer_size
                                   nl, ... %num_labels
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn, hl, nl, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
t1 = reshape(nn(1:hl * (il + 1)), ...
                 hl, (il + 1));

t2 = reshape(nn((1 + (hl * (il + 1))):end), ...
                 nl, (hl + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(t1));
Theta2_grad = zeros(size(t2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Squared terms to be used in regularization part of J:
squared_1 = t1.^2;
squared_2 = t2.^2;

%forwardpropagation steps:
a1 = [ones(m, 1) X];
h1 = sigmoid(a1 * t1');
a2 = [ones(m, 1) h1];
h2 = sigmoid(a2 * t2');
a3 = h2;
yv = [1:nl] == y;
%X_h1 = [ones(m, 1) h1];

%backpropagation:
d_3 = a3 - yv;
d_2 = (d_3*t2(:,2:end)) .* sigmoidGradient(a1*t1');
delta_2 = d_3'*a2; 
delta_1 = d_2'*a1;

%Cost function:
J = 1/m * sum(sum(-yv.*log(h2) - (1-yv).*log(1-h2))) + ...
    lambda/(2*m)*(sum(sum(squared_1(:,2:end))) + sum(sum(squared_2(:,2:end))));

%Gradient calculation:
t1(:,1) = 0;
t2(:,1) = 0;
t1 = t1 * (lambda/m);
t2 = t2 * (lambda/m);
Theta1_grad = 1/m * delta_1 + t1;
Theta2_grad = 1/m * delta_2 + t2;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
