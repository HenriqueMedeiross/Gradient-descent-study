# Gradient-descent

Implementation of the gradient descent algorithm for polynomial regretion.

The gradient descent is based on the fact that we can minimize some cost function J using the negative gradient of this function achieving your local minimum.

Lets get a few steps back... What exactly we are trying to minimize?

Say you have some function y = ax+b where the x is your entry data and the y is your expected result. Let's suppose that the data is distributed like this:

<img src="https://github.com/HenriqueMedeiross/Gradient-descent-study/blob/master/Equations/scatt.PNG?raw=true">

You can easily guess some values for a and b and find some line that roughly fit in those points, but when you need to be more accurate and versatile this work gets a little harder.

To get a little more generic we'll now assume that we have *n* features (entries or "x") to produce 1 output *y*. Besides that we also have *m* training examples. Now combine all of this in a matrix/vector form.

The entry <img src="https://github.com/HenriqueMedeiross/Gradient-descent-study/blob/master/Equations/x^m_n.png?raw=true"> vector:
<img src="https://github.com/HenriqueMedeiross/Gradient-descent-study/blob/master/Equations/x_matrix.png?raw=true">

The output *y* vector: <img src="https://github.com/HenriqueMedeiross/Gradient-descent-study/blob/master/Equations/y_matrix.png?raw=true">

After this, get all the *m* vectors x and transpose and stack them like this: <img src="https://github.com/HenriqueMedeiross/Gradient-descent-study/blob/master/Equations/X-matrix.png?raw=true">

Unlike the function y = ax+b, we have more than 1 entry *x*, we have *n* entries (that was written above as a vector). Consider a and b as weigths of that function y, now if we have *n* features we will need *n+1* weigths to construct some linear model function. Those weigths are called theta, and it's vector form is as follows:

<img src="https://github.com/HenriqueMedeiross/Gradient-descent-study/blob/master/Equations/theta_vector.png?raw=true">

With all of this in mind, it's now possible to elaborate a hypothesis function, that is, the function we will use to predict the y values.


<img src="https://github.com/HenriqueMedeiross/Gradient-descent-study/blob/master/Equations/h(x).png?raw=true">


So, for each training exemple m, we will predict it's value and compare with the correct one (y). To do this we'll finaly need to get a cost function J:
 
<img src="https://github.com/HenriqueMedeiross/Gradient-descent-study/blob/master/Equations/costfunc.png?raw=true">

What this function does is compare the predicted value h(x_i) with the correct value y_i (where *i* is the current training example), get the square of this result and sum with all the other training examples. This function J is what we need to minimize in order to find the right weigths that fit in our dataset.

As you can see, our actual variables are not the X's, but the theta vector (i.e. what we will variate untill we find the best fit) so to minimize the J(theta) function we need to find it's gradient (that tells us the direction of the function growth) and scale it, then subtract the correspondent gradient of each weigth and subtract it (to minimize) from the respective theta. The formula of this operation is as follows:

<img src="https://github.com/HenriqueMedeiross/Gradient-descent-study/blob/master/Equations/update%20weights%20function.png?raw=true">






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
