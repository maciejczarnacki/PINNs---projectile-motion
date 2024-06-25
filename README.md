## Simple example of PINNs usage - projectile motoion simulation

### 1. Introduction

In this repository I would like to present one of the simplest example of "Physics Informed Neural Networks - PINNs" usage. This is modeling projectile motion in the uniform earth's gravitational field. I will not explain here what exactly PINNs is, but will only focus on discussing the presented code and presenting the results. At the end, I provide links to the most interesting sources of knowledge on this topic.

To run the code in Jupyter Notebook, you need to install the PyTorch package for constructing and training neural networks and Matplotlib for visualizing the obtained results.

### 2. Projectile motion

There's basically nothing to write here. The task is very simple. We consider the motion of a body in a uniform gravitational field. 
We can distinguish several cases depending on the angle of inclination of the initial velocity vector to the direction of motion of the thrown body.
Example equations can be found e.g. here https://en.wikipedia.org/wiki/Projectile_motion

This task involves solving the following system of equations:

$$ \dfrac{d^2y}{dt^2} - g = 0 $$

$$ \dfrac{d^2x}{dt^2} = 0 $$

with appropraite initial conditions y(0) = h_0, x(0) = 0.

### 3. The code

Necessary packages import

```python
import torch
from torch import nn
import matplotlib.pyplot as plt

from PIL import Image
```

Function definition of exact mathematical solution. This is required for collecting training and validation data.
```python
def projectile_motion_gen(t, g, v_0, h_0, alpha):
    x = v_0 * torch.cos(alpha) * t
    y = h_0 + v_0 * torch.sin(alpha) * t + 0.5 * g * t**2
    return x.view(-1,1), y.view(-1,1)
```

Training data genration, 
where g - gravitational acceleration (standard gravity), 
v_0 - initial velocity, 
h_0 - initial heigh, alpha - angle between v_0 velocity vector and x-axis.

```python
g = -9.81
v_0 = 6.5
h_0 = 1.5
alpha = torch.Tensor([torch.pi/3]) # pi/3 -> 60 degrees

# time space definition
t = torch.linspace(0, 1.5, 100).view(-1,1)

# collecting data
x, y = projectile_motion_gen(t, g, v_0, h_0, alpha)

# choosing data for NN training, 5 points
x_data = x[0:40:8]
y_data = y[0:40:8]
t_data = t[0:40:8]

```

Training data and exact mathematical solution visualization.

```python
plt.figure()
plt.plot(x, y, label='mathematical solution')
plt.scatter(x_data, y_data, color='red', label='training data')
plt.legend()
plt.title('Training data')
plt.show()
```

![Data set](/img/data_set.png)

Neural network class definition. I have used simple linear structure consist of input layer with one neuron, four hidden layers with 32 neurons each,
output layer with 2 neurons. For all hidden layers hyperbolic tangent activation function was used.

```python
class PM(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()
        self.input_layer = nn.Linear(n_input, n_hidden)
        self.hidden_layer_1 = nn.Linear(n_hidden, n_hidden)
        self.hidden_layer_2 = nn.Linear(n_hidden, n_hidden)
        self.hidden_layer_3 = nn.Linear(n_hidden, n_hidden)
        self.hidden_layer_4 = nn.Linear(n_hidden, n_hidden)
        self.output_layer = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.input_layer(x)
        x = torch.tanh(x)
        x = self.hidden_layer_1(x)
        x = torch.tanh(x)
        x = self.hidden_layer_2(x)
        x = torch.tanh(x)
        x = self.hidden_layer_3(x)
        x = torch.tanh(x)
        x = self.hidden_layer_4(x)
        x = torch.tanh(x)
        output_ = self.output_layer(x)
        return output_
```

Neural network object declaration and optimizer choose. In this step it has to be define time space for physics loss optimization.

```python
model = PM(1, 2, 32)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

t_physics = torch.linspace(0, 1.5, 20).view(-1, 1).requires_grad_(True)

```

Learning loop definition.

```python
epochs = 7000
losses_p = []
losses_d = []
files = []
for i in range(epochs):
    optimizer.zero_grad()
    
    # forward pass of training data and loss calculation
    z = model(t_data)
    x_h, y_h = z[:,[0]], z[:,[1]]
    loss_1 = torch.mean((x_h - x_data)**2) + torch.mean((y_h - y_data)**2)

    # forward pass for physics - gradients calculation and physics loss estimation
    z_p = model(t_physics)
    x_h_p, y_h_p = z_p[:,[0]], z_p[:,[1]]
    dx = torch.autograd.grad(x_h_p, t_physics, torch.ones_like(x_h_p), create_graph=True)[0]
    d2x = torch.autograd.grad(dx, t_physics, torch.ones_like(dx), create_graph=True)[0]
    dy = torch.autograd.grad(y_h_p, t_physics, torch.ones_like(y_h_p), create_graph=True)[0]
    d2y = torch.autograd.grad(dy, t_physics, torch.ones_like(dy), create_graph=True)[0]

    physics_x = d2x - 0
    physics_y = d2y - g

    loss_x = torch.mean(physics_x**2)
    loss_y = torch.mean(physics_y**2)
    loss_physics = loss_x + loss_y
    loss = loss_1 + 0.00075 * loss_y + 0.00075 * loss_x

    # backpropagation
    loss.backward()
    optimizer.step()

    losses_p.append(loss_physics.item())
    losses_d.append(loss_1.item())

    # png files generation
    if (i + 1) % 25 == 0:
        z_p = model(t).detach()
        x_h_p, y_h_p = z_p[:,[0]], z_p[:,[1]]
        
        # plot_in_run(x, y, x_h_p, y_h_p, x_data, y_data, i)
        plots_in_run_xyt(t, t_data, x, y, x_h_p, y_h_p, x_data, y_data, i, losses_p, losses_d)

        file = "plots/pinn_%.8i.png"%(i+1)
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        files.append(file)

        if (i) % 1000 == 0: plt.show()
        else: plt.close("all")

# Animated gif generation
save_gif_PIL("pinn.gif", files, fps=20, loop=0)
```

The most important part of PINNs algorithm is automatic gradient calculation via PyTorch, Tensorflow or JAX.

Main difference between "normal" neural network training and PINNs is loss function construction.
Loss contains two parts, first from data and second from differential equations describing physical system.
That physical loss is some kind of regularization or additional constrains for neural network optimization.


### 4. Results

a) Learning from data points with known differential equation parameter g.

One picture is worth a thousand words.
Just look at the animation below.
The following four charts show steps of the learning loop. The course of the solutions x(t), y(t) and the motion trajectory of y(x) are shown. 
The last graph shows the cost function values ​​for data and physics.

![PINN](/img/pinn.gif)

b) Learning from data points without known differential equation parameter g - discovery mode.

In this section i want to show simple reasemble of this experiment.
Suppose we want to determine the gravitational acceleration (standard gravity). Our data for training a neural network comes from measurement, not from an exact mathematical solution. By indicating in the program code that the algorithm should optimize the parameter g in the equation of motion, we can obtain its value while training the neural network.

Below I present the necessary changes to the code.

```python
g = nn.Parameter(torch.zeros(1, requires_grad=True))

optimizer = torch.optim.Adam(list(model.parameters())+[g], lr=0.0005)
```

Now our model needs more steps to converge. Change is from 7000 to 32000.
This is understandable because the optimizer has "more work to do".


![PINN](/img/pinn_g.gif)

### 5. Literature

https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/

https://youtu.be/G_hIppUWcsc?si=zz58voDH4lObcGOL

https://maziarraissi.github.io/PINNs/

https://medium.com/@theo.wolf/physics-informed-neural-networks-a-simple-tutorial-with-pytorch-f28a890b874a