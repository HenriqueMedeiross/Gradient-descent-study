import matplotlib.pyplot as plt
import numpy as np
import random

# Just initialization of some set of values, skip to line 64
def coord():
    return np.array([(1.38,1.13),
    (2.52,2.47),
    (4.06,2.61),
    (5.06,1.41),
    (5,1.11),
    (3.74,1.27),
    (3.32,1.81),
    (3.9,0.67),
    (4.8,2.13),
    (4.84,2.55),
    (5.42,3.37),
    (6.54,3.71),
    (7.1,2.93),
    (3.24,0.97),
    (6.76,2.73),
    (6.22,2.77),
    (6.54,1.79),
    (7.32,2.23),
    (6.82,4.99),
    (6.04,5.25),
    (5.22,5.1),
    (3.98,3.61),
    (4.2,3.95),
    (8.94,6.91),
    (9.22,5.65),
    (9.78,5.39),
    (8.14,5.71),
    (7.88,6.53),
    (8.26,7.85),
    (9.46,8.85),
    (10.84,8.91),
    (12.7,7.95),
    (14.1,6.77),
    (13.72,5.95),
    (11.86,7.51),
    (9.92,7.23),
    (2.26,11.97),
    (16.56,0.19),
    (10.6,1.13),
    (3.36,7.49),
    (11.26,6.87),
    (10.86,7.91),
    (12.9,9.49),
    (14.58,10.73),
    (16.22,9.49),
    (14.82,8.93),
    (13.38,8.29),
    (13.28,10.95),
    (11.32,10.25),
    (11.22,11.1),
    (6.28,8.67),
    (9.32,9.67),
    (12.08,9.95),
    (14.36,11.93),
    (15.62,12.11),
    (16.48,13.11),
    (17.18,13.41)])

coordinates = coord()
x_val = [x[0] for x in coordinates]
y_val = [y[1] for y in coordinates]
t = np.linspace(0,np.max(x_val))

# Separate the training set evaluated by n (0.0 to 1.0), and the rest of the dataset will be the test set
def training_test_set(n): 
    aleatory_training_samples = random.sample(range(0,len(x_val)),int(n*len(x_val)))
    aleatory_test_samples = np.delete(np.arange(0,len(x_val)),aleatory_training_samples)
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in aleatory_training_samples:
        x_train.append(x_val[i])
        y_train.append(y_val[i])
    for i in aleatory_test_samples:
        x_test.append(x_val[i])
        y_test.append(y_val[i])
    return np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test)

n = 0.75 # n = 0.75 means that 75 percent of the dataset will be used for training
training_X_values,training_Y_values,test_X_values,test_Y_values = training_test_set(n) 

# Inicialization hypothesis terms theta0 and theta1
th0 = np.random.random()
th1 = np.random.random() 
alpha = 0.01 # Learning rate
iteractions = 0
max_iteractions = 100000 # Limit of iteractions
error_list = []

# Gradient descent
while iteractions < max_iteractions:
    m = len(training_X_values)
    # Hypothesis function : h(x) = th0 + th1*x.
    # In this case the x is the training_X_values vector, and the h will be the predicted values vector
    h = th0 + th1 * training_X_values 

    # Here we take the half of the mean square error
    mean_square_error = np.sum((h-training_Y_values)**2)/(2*m) 
    
    # Error list to plot the error graph at the end
    error_list.append(mean_square_error) 

    # The mean_square_error derivatives, J(theta) = (1/2m) * sum((th0 - th1*X))^2 so we take the partial derivative of this functions related to th0, then to th1
    prime_th0 = np.sum(h-training_Y_values)/m
    prime_th1 = np.sum((h-training_Y_values)*training_X_values)/m

    # Update the values of th0 and th1 based on the negative value of the partial derivative, so that we can minimize the cost function J
    th0 = th0 - alpha*prime_th0
    th1 = th1 - alpha*prime_th1
    
    # If the error is already too small we can stop the loop
    iteractions+=1
    if iteractions > 3:
        if ((error_list[-1] - error_list[-2])**2)**(1/2) < 10**-15:
            break

# %%
# Plots to visualize the results of the error
print(iteractions)
#.plot(t,th0+th1*t)
plt.scatter(x_val,y_val)
plt.show()
plt.plot(np.arange(len(error_list)),error_list)