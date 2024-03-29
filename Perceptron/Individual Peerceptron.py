from pylab import *


def perceptronUpdate(x, y, w, b):
    n, d = x.shape
    w0 = w
    b0 =b

    mark = np.sign(x.dot(w0) + b0)
    print("initial label is \n", mark, "\n")
    while (y != mark).all:
        for i in range(0,n):
            print("y",i, "is\t\t\t", y[i])
            marki = np.sign((sum(w * x[i]) + b))
            print("label", i, "is: \t", marki)
            if y[i] * marki <= 0:
                print("label does not equal to y, we need to update weights")
                print("waights are: \t\t", w)
                w += y[i] * x[i]
                print("new weight are: \t", w)
                b += y[i]
                print("new b is: \t\t\t", round(b,2))
                marki_new = np.sign((sum(w * x[i]) + b))
                print("new label is: \t\t",marki_new, "\n")

            else:
                print("OK!\n")

        mark = np.sign(x.dot(w) + b)
        print("updated label is ", mark, "\n")

        if (y == mark).all():
            break
        else:
            continue

    return (w,b,mark)


"""xs = np.zeros((4,2))             #midterm exam
xs[0] = [1.,1.]
xs[1] = [-1.,1.]
xs[2] = [-1.,-1.]
xs[3] = [1.,-1.]

ys = np.zeros((4))
ys = [1., 1., -1., 1.]
ys = np.array(ys)

w0 = np.zeros((2))
w0 = [0.9, 0.5]

b0 = -0.2"""
xs = np.zeros((4,2))                #class slides
xs[0] = [1.,1.]
xs[1] = [1.,-1.]
xs[2] = [-1.,1.]
xs[3] = [-1.,-1.]

ys = np.zeros((4))
ys = [1., -1., -1., -1.]

w0 = np.zeros((2))
w0 = [0.6, 0.2]

b0 = -0.9
print("xs is \n", xs, "\n")
print("w0 is \t", w0)
print("b0 is \t", b0)
print("ys is \t", ys, "\n")

w,b,yf= perceptronUpdate(xs, ys, w0, b0)

print("final b is: \t", np.around(b, decimals=2))
print("final w is: \t", w)
print("final label is: ", yf)

