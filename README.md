# Gradient-descent

Implementation of the gradient descent algorithm for polynomial regretion.

The gradient descent is based on the fact that we can minimize some cost function J using the negative gradient of this function achieving your local minimum.

Lets get a few step back... What exactly we are trying to minimize?

Say you have some function y = ax+b where the x is your entry data and the y is your expected result. 





<img src="https://github.com/HenriqueMedeiross/Gradient-descent-linear-regretion-1-variable/blob/master/Equations/eq1.png?raw=true" width="30%" height="30%">

where i is the index of the sample

The mean square error function J is as follows:

<img src="https://github.com/HenriqueMedeiross/Gradient-descent-linear-regretion-1-variable/blob/master/Equations/eq2.png?raw=true" width="30%" height="30%">

To minimize the error function J we use the gradient descent <img src="https://github.com/HenriqueMedeiross/Gradient-descent-linear-regretion-1-variable/blob/master/Equations/eq3.png?raw=true" width="10%" height="10%">:

<img src="https://github.com/HenriqueMedeiross/Gradient-descent-linear-regretion-1-variable/blob/master/Equations/eq4.png?raw=true" width="30%" height="30%">

<img src="https://github.com/HenriqueMedeiross/Gradient-descent-linear-regretion-1-variable/blob/master/Equations/eq5.png?raw=true" width="30%" height="30%">

In this script I used the numpy libreary that makes a lot easyer to make this vector calculations.

\**Please notify me if you find some error or if there is any how to improve my code*

$x^i_n$ where $i$ is the number of training examples and $n$ is the number of features, this means that $x^i$ is a vector with $n$ features


$h(x^m) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n$
