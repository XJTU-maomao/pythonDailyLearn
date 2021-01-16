import torch
from torch.autograd import Variable

# Variable in torch is to build a computational graph,
# but this graph is dynamic compared with a static graph in Tensorflow or Theano.
# So torch does not have placeholder, torch can just pass variable to the computational graph.

tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor, requires_grad = True)

print(tensor)
print(variable)

# till now the tensor and variable seem the same.
# However, the variable is a part of the graph, it's a part of the auto-gradient.
t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)
print(t_out)
print(v_out)

v_out.backward()
# backpropagation from v_out
# v_out = 1/4 * sum(variable*variable)
# the gradients w.r.t the variable, d(v_out)/d(variable) = 1/4*2*variable = variable/2
print(variable.grad)

print(variable)   # this is data in variable format

print(variable.data)  # this is data in tensor format

print(variable.data.numpy())  # numpy format
