learnableReluLayer.m and widerLearnableReluLayer.m are the two implementation of Mexican ReLU.
Since they are nearly equivalent, here only learnableReluLayer is considered.

They are subclasses of the class nnet.layer.Layer

learnableReluLayer(numChannels, name, learnRateFactor, maxInput) is the constructor of the class.
It requires the number of channels to be specified in advance.

predict(layer,X) computes the forward pass. It is the sum of a PReLU activation and three Mexican hat functions.
The Mexican hat function referred to Beta has negative values, while the others are positive.

backward(layer,X,dLdZ) computes the backward pass. The gradient of the loss L with respect to Z is calculated
using the chain rule dLdX = dLdZ * dZdL. Z is the output of the MeLU layer and dLdZ is the gradient of the loss
with respect to the output. We compute dZdX, which is the gradient of the output w.r.t the input.

dZdX is iteratively computed using the chain rule. In our case Z = predict(X,alpha,beta,gamma,delta).
We omit the dependence on alpha, beta, gamma and delta in the calculation of dLdX. Let Z = f(X).
From the formula in predict we know that

Z = m*g(X/m),

where m is maxInput and g(x) = f(m*x)/m.

Z' := dZdX = d(m*g(X/m))dX = m*d(g(X/m))dX = m*(1/m)g'(X/m) = g'(X/m)

by the chain rule, where g' stands for the derivative of g.
The derivative of g is easy to calculate since it the sum of the derivatives of PReLU and the Mexican hat functions, multiplied by their parameters.
The derivative of PReLU is alpha for x < 0 and 1 for x > 0.
The derivative of a Mexican hat function M of parameters a and lambda is

M'(x) = +1 for     a - lambda < x < a
M'(x) = -1 for     a < x < a + lambda

and 0 everywhere else.

g'is calculated by iteratively overwriting its values. Between 1 and 2, for instance, its derivative is given by the sum of
1, which is the derivative of PReLU in that interval,
- beta, which is the derivative of the first Meican function in that interval (negative sign because it is the opposite of a "true" mexican function) -
- gamma, which is the derivative of the third Mexican function in that interval
= 1 - beta - gamma

The derivatives are calculated in different steps and are iteratively overwritten.

The derivatives of Z with respect to alpha, beta, gamma and delta are easy to calculate.

