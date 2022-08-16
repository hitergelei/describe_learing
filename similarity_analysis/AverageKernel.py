from time import time
from dscribe.descriptors import SOAP
from dscribe.kernels import AverageKernel

from ase.build import bulk
from ase.build import molecule

# We will compare two similar molecules
# a = molecule("H2O")
# b = molecule("H2O2")
a = bulk("Al", "fcc", a=4.05)
b = bulk("Al", "bcc", a=3.2)


# First we will have to create the features for atomic environments. Lets use SOAP.
desc = SOAP(species=["Al"], rcut=5.0, nmax=2, lmax=2, sigma=0.2, periodic=False, crossover=True, sparse=False)

# start_time = time()
a_features = desc.create(a)
b_features = desc.create(b)
# end_time = time()
# print("total time: {} s".format(end_time - start_time))
print("-----------------")
print(a_features)
print(b_features)
print("------------------")


# Calculates the similarity with an average kernel and a linear metric. The
# result will be a full similarity matrix.
re = AverageKernel(metric="linear")
re_kernel = re.create([a_features, b_features])
print("re_kernel = \n")
print(re_kernel)
print("----------------------------------------")

# Any metric supported by scikit-learn will work: e.g. a Gaussian:
re = AverageKernel(metric="rbf", gamma=1)
re_kernel = re.create([a_features, b_features])
print("re_kernel = \n")
print(re_kernel)
"""
for molecules
re_kernel = 
array([[1.        , 0.99352111],
       [0.99352111, 1.        ]])

---------------------      
for Al
re_kernel = 

[[1. 1.]
 [1. 1.]]
"""