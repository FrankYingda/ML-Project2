import numpy as np
from pylab import *

# <GRADED>
def classifyLinear(xs, w, b):
    """
    function preds=classifyLinear(xs,w,b)

    Make predictions with a linear classifier
    Input:
    xs : n input vectors of d dimensions (nxd) [could also be a single vector of d dimensions]
    w : weight vector of dimensionality d
    b : bias (scalar)

    Output:
    preds: predictions (1xn)
    """
    w = w.flatten()
    #predictions = np.zeros(xs.shape[0])

    ## fill in code ...
    predictions = np.sign(xs.dot(w) + b)

    ## ... until here
    print("predictions: ", predictions)
    return predictions
# </GRADED>

# test classifyLinear code:
xs=rand(5,3)-0.5 # draw random data
w0=np.array([0.5,-0.3,0.4]) # define a random hyperplane
b0=-0.1 # with bias -0.1
ys=np.sign(xs.dot(w0)+b0) # assign labels according to this hyperplane (so you know it is linearly separable)
print("xs is ", xs)
print("w0 is ", w0)
print("b0 is ", b0)
print("ys is ", ys)

assert (all(np.sign(ys*classifyLinear(xs,w0,b0))==1.0))  # the original hyperplane (w0,b0) should classify all correctly
print("Looks like you passed the classifyLinear test! :)")

