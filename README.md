# Gradient-descent-linear-regretion-1-variable

Implementation of the gradient descent algorithm for linear regretion with 1 variable in python.

The hypothesis function that we want to achieve/predict:
img src="https://render.githubusercontent.com/render/math?math=h(x^i) = \theta_0+\theta_1x^i
where $i$ is the index of the sample

The mean square error function J is as follows:
img src="https://render.githubusercontent.com/render/math?math=J(\theta_0,\theta_1) = \frac{\sum(h(x^i)-y^i)^2}{2m}

To minimize the error function $J$ we use the gradient descent img src="https://render.githubusercontent.com/render/math?math=-\bigtriangledown J(\theta):

$\theta_0 = \theta_0 - \alpha \frac{\partial}{\partial\theta_0} J(\theta_0,\theta_1)$

$\theta_1 = \theta_1 - \alpha \frac{\partial}{\partial\theta_1} J(\theta_0,\theta_1)$

In this script I used the numpy libreary that makes a lot easyer to make this vector calculations.
