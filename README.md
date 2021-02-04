# Gradient-descent

Implementation of the gradient descent algorithm for polynomial regretion.

The gradient descent is based on the fact that we can minimize some cost function J using the negative gradient of this function achieving your local minimum.

Lets take a few steps back... What exactly we are trying to minimize?

Say you have some function y = ax+b where the x is your entry data and the y is your expected result. Let's suppose that the data is distributed like this:

<img src="https://github.com/HenriqueMedeiross/Gradient-descent-study/blob/master/Equations/scatt.PNG?raw=true">

You can easily guess some values for a and b and find a line that roughly fit in those points, but when you need to be more accurate and versatile this work gets a little harder.

To get a little more generic we'll now assume that we have *n* features (entries or "x") to produce 1 output *y*. Besides that we also have *m* training examples. Now combine all of this in a matrix/vector form.

The entry <img src="https://github.com/HenriqueMedeiross/Gradient-descent-study/blob/master/Equations/x^m_n.png?raw=true"> vector:
<img src="https://github.com/HenriqueMedeiross/Gradient-descent-study/blob/master/Equations/x_matrix.png?raw=true">

The output *y* vector: <img src="https://github.com/HenriqueMedeiross/Gradient-descent-study/blob/master/Equations/y_matrix.png?raw=true">

After this, get all the *m* vectors x and transpose and stack them like this: <img src="https://github.com/HenriqueMedeiross/Gradient-descent-study/blob/master/Equations/X-matrix.png?raw=true">


Unlike the function y = ax+b, we have more than 1 entry *x*, we have *n* entries (that was written above as a vector). Consider a and b as weigths of that function y, now if we have *n* features we will need *n+1* weigths to construct some linear model function. Let's call those theta, and it's vector form is as follows:

<img src="https://github.com/HenriqueMedeiross/Gradient-descent-study/blob/master/Equations/theta_vector.png?raw=true">

With all of this in mind, it's now possible to elaborate a hypothesis function, that is, the function we will use to predict the y values.


<img src="https://github.com/HenriqueMedeiross/Gradient-descent-study/blob/master/Equations/h(x).png?raw=true">

You may notice that we have n features and n+1 weigths, as we'll further vectorize this calculations, for convention it needs to be added in the first column of the X matrix, a "ones column" that will represent the *x_0*, and as you can imagine, the results will not be changed because it will multiply theta_0, and that weigth has no feature attached to it

So, for each training exemple m, we will calculate the error produced by our theta vector, getting the predicted value and compare(subtract) from the correct one (y). To do this we'll finaly need to get a cost function J, in this case, the mean square error function:
 
<img src="https://github.com/HenriqueMedeiross/Gradient-descent-study/blob/master/Equations/costfunc.png?raw=true">

What this function does is compare the predicted value <img src="https://github.com/HenriqueMedeiross/Gradient-descent-study/blob/master/Equations/h(x^i).png?raw=true"> with the correct value <img src="https://github.com/HenriqueMedeiross/Gradient-descent-study/blob/master/Equations/y^i.png?raw=true"> (where *i* is the current training example), get the square of this result and sum walking through all the other training examples. This function J is what we need to minimize in order to find the right weigths that has the best fit in our dataset.
</br>
As you can see, our actual variables are not the X's, but the theta vector (i.e. what we will variate untill we find the best fit) so to minimize the J(theta) function we need to find it's gradient (that tells us the direction of the function growth) and scale it  (  <img src="https://github.com/HenriqueMedeiross/Gradient-descent-study/blob/master/Equations/frac{1}{2m}.png?raw=true" heigth="2.5%" width="2.5%">  )  , then subtract the correspondent gradient of each weigth and subtract it (to minimize) from the respective theta. The formula of this operation is as follows:

<img src="https://github.com/HenriqueMedeiross/Gradient-descent-study/blob/master/Equations/update%20weights%20function.png?raw=true">

*Please remember that theta isn't a scalar, but a vector instead*

If you have some calculus knowledge can easily conclude that the partial derivative of the function J related to theta_n is:


<img src="https://github.com/HenriqueMedeiross/Gradient-descent-study/blob/master/Equations/gradient-func-J.png?raw=true">


Now our gradient descent is ready to work, but it can be less computational expensive without all those sums. In order to vectorize the h(x), we can simply multiply X with theta since X's dimentions are (m x n) and theta's are (n x 1), so our h matrix will result like y, (m x 1) as expected.

The function J (mean square error) vectorized can be written like this:  

<img src="https://github.com/HenriqueMedeiross/Gradient-descent-study/blob/master/Equations/mean-square-vect.png?raw=true">

To update the weights in vectorized way you can transpose the X matrix(getting a (n x m)) and multiply it by the error vector (h-y)^2 with (m x 1) dimentions and the resoult will be a (n x 1) vector, just like theta:

<img src="https://github.com/HenriqueMedeiross/Gradient-descent-study/blob/master/Equations/update-weights-function-vect.png?raw=true">


*Please let me know if you find some error*