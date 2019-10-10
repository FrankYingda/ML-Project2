import numpy as np
from matplotlib import *
#matplotlib.use('PDF')
from pylab import *

#</GRADED>
import sys
import matplotlib.pyplot as plt
import time

# add p02 folder
#sys.path.insert(0, 'K:\GWU\ML\Project2')

#%matplotlib notebook
#print('You\'re running python %s' % sys.version.split(' ')[0])

from retry import retry
@retry(tries=20, delay=2)
# <GRADED>
def test():
    def perceptronUpdate(x, y, w):
        """
        function w=perceptronUpdate(x,y,w);

        Implementation of Perceptron weights updating
        Input:
        x : input vector of d dimensions (d)
        y : corresponding label (-1 or +1)
        w : weight vector of d dimensions

        Output:
        w : weight vector after updating (d)
        """
        assert (y in {-1, 1}), "the number of y is invalid"
        assert (len(w.shape) == 1), "At the update w must be a vector not a matrix (try w=w.flatten())"
        assert (len(x.shape) == 1), "At the update x must be a vector not a matrix (try x=x.flatten())"

        ## fill in code ...
        Mark = 0
        while y != Mark:
            print("waights are: \t\t", w)
            p = w*x
            print("vector x*w is \t\t", p)
            Sum1 = sum(p)
            print("sum of vector is: \t", Sum1)

            if Sum1 >= 0:
                Mark = 1
                print("label is: \t\t\t", Mark)
                print("y is: \t\t\t\t", y)
                if y == -1:
                    print("so we need to update waights\n")
                    w -= x
                else:
                    print("OK!\n")
                    break
            elif Sum1 < 0:
                Mark = -1
                print("label is: \t\t\t", Mark)
                print("y is: \t\t\t\t", y)
                if y == 1:
                    w += x
                    print("so we need to update waights\n")
                else:
                    print("OK!\n")
                    break

        ## ... until here
        print("the final waights are: ", w.flatten())
        return w.flatten()
# </GRADED>

    # test the update code:
    x=rand(5) # random weight vector
    w=rand(5) # random feature vector
    y=-1 # random label

    print("initializing x: \t", x)
    print("initializing y: \t", y)
    #print("initializing w: \t", w.copy())

    wnew=perceptronUpdate(x,y,w.copy()) # do a perceptron update
    print("norm is: \t\t\t\t", norm(wnew-w+x),"\n")
    assert(norm(wnew-w+x)<1e-10), "perceptronUpdate didn't pass the test : (" # if correct, this should return 0
    print("Looks like you passed the update test : )\n")

test()
