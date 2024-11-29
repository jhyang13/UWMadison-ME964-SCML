#import the needed packages
import deepxde as dde
#the DeepXde package
import matplotlib.pyplot as plt
#the package for plotting figures
import numpy as np
#a fundatmental package for scientific computing in Python
from deepxde.backend import tf
#the deep learning framework will be used, you can choose other backend like pytorch
# Import torch if using backend pytorch
# import torch


#load the data in the package
def gen_testdata():
    data = np.load("dataset/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y


#define the PDE
def pde(x, y):
    # hessian is the second order derivative
    # jacobian is the first order derivative
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    # 0.01 is the viscosity, Re=1/\nu=100 in this case
    return dy_t + y * dy_x - 0.01 / np.pi * dy_xx


#define the spatial domain for PDE
geom = dde.geometry.Interval(-1, 1)
#define the time domain for PDE
timedomain = dde.geometry.TimeDomain(0, 0.99)
#generate the whole domain for PDE
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


# generater the boundary condition and initial condition
bc = dde.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.IC(
    geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial
)
# geomtime/pde/[bc,ic] have been defined
#num_domain is the sampled pts for train process in the whole domain
#num_boundary is the sampled pts for training process on the boundary
#num_initial is the sampled pts for training process on the initial
data = dde.data.TimePDE(
    geomtime, pde, [bc, ic], num_domain=2540, num_boundary=80, num_initial=160
)


#define the structure and hyperparameters of the neural network
#input layer 2 neuron, 3 hidden layers with 20 neurons and 1 output layer with 1 neruon
net = dde.maps.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)
# complie the model using 'adam' optimizer with 10e-3 initial learning rate
model.compile("adam", lr=1e-3)
# using adam to train 15k epochs firstly
model.train(epochs=15000)
# than use the 'L-BfGS' to train until the algorithm determines convergence
model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=True, isplot=True)


#test data
X, y_true = gen_testdata()
y_pred = model.predict(X)
f = model.predict(X, operator=pde)
print('Mean residual:', np.mean(np.absolute(f)))
print('L2 relative error:', dde.metrics.l2_relative_error(y_true, y_pred))
np.savetxt('test.dat', np.hstack((X, y_true, y_pred)))



