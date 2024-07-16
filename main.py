
# perceptron

def dot(v1: list,v2: list) -> float:
    return sum([x*y for x, y in zip(v1,v2)])

assert dot([1,2,3],[4,5,6]) == 32
# 1 * 4 + 2 * 5 + 3 * 6 = 4 + 10 + 18 = 32

def step_func(x):
    return 1 if x >= 0 else 0

def perceptron_output(weights: list, bias: float, x: list) -> float:
    res = dot(weights, x) + bias
    return step_func(res)

weights = [2,2]
bias = -3

assert perceptron_output(weights,bias,[1,1]) == 1
assert perceptron_output(weights,bias,[0,1]) == 0
assert perceptron_output(weights,bias,[1,0]) == 0
assert perceptron_output(weights,bias,[0,0]) == 0

weights = [2,2]
bias = -1

assert perceptron_output(weights,bias,[1,1]) == 1
assert perceptron_output(weights,bias,[0,1]) == 1
assert perceptron_output(weights,bias,[1,0]) == 1
assert perceptron_output(weights,bias,[0,0]) == 0

weights = [-2]
bias = 1
assert perceptron_output(weights,bias,[1]) == 0
assert perceptron_output(weights,bias,[0]) == 1

# feedforward neural networks

import math

def sigmoid(t: float) -> float:
    return 1 / (1 + math.exp(-t))

def neuron_output(weights: list, inputs: list) -> float:
    return sigmoid(dot(weights, inputs))

def feed_forward(neural_network: list, input_vector: list) -> list:
    outputs: list = []
    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias) for neuron in layer]
        outputs.append(output)
        input_vector = output
    return outputs

xor_network = [
    [[20,20,-30], 
     [20,20,-10]],

     [[-60,60,-30]]
]
assert 0.000 < feed_forward(xor_network, [0, 0])[-1][0] < 0.001
assert 0.999 < feed_forward(xor_network, [1, 0])[-1][0] < 1.000
assert 0.999 < feed_forward(xor_network, [0, 1])[-1][0] < 1.000
assert 0.000 < feed_forward(xor_network, [1, 1])[-1][0] < 0.001

# backpropagation

def sqerror_gradients(network: list, input_vector: list, target_vector: list) -> list:

    hidden_outputs, outputs = feed_forward(network, input_vector)

    output_deltas = [output * (1 - output) * (output - target)
                     for output, target in zip(outputs, target_vector)]


    output_grads = [[output_deltas[i] * hidden_output
                     for hidden_output in hidden_outputs + [1]]
                    for i, output_neuron in enumerate(network[-1])]

    
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                         dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]

    
    hidden_grads = [[hidden_deltas[i] * input for input in input_vector + [1]]
                    for i, hidden_neuron in enumerate(network[0])]

    return [hidden_grads, output_grads]

import random
random.seed(0)

# train data
xs = [[0,0],[0,1],[1,0],[1,1]]
ys = [[0],[1],[1],[0]]

# start with random weights
network = [
    [[random.random() for _ in range(2 + 1)],
     [random.random() for _ in range(2 + 1)]],
     [[random.random() for _ in range(2 + 1)]]
]

def scalar_multiply(s: float, l: list) -> list:
    return [s * v for v in l]

assert scalar_multiply(2,[1,2,3]) == [2,4,6]

def add(l1: list, l2: list) -> list:
    return [x + y for x, y in zip(l1,l2)]

assert add([1,2,3],[4,5,6]) == [5,7,9]

def gradient_step(v: list, gradient: list, step_size: float) -> list:
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

import tqdm # pip install tqdm

learning_rate = 1.0

for epoch in tqdm.trange(20000, desc='neural network for xor'):
    for x, y in zip(xs, ys):
        gradients = sqerror_gradients(network, x, y)
        network = [[gradient_step(neuron, grad, -learning_rate) for neuron, grad in zip(layer, layer_grad)] for layer, layer_grad in zip(network, gradients)]
    
assert feed_forward(network, [0,0])[-1][0] < 0.01
assert feed_forward(network, [0,1])[-1][0] > 0.99
assert feed_forward(network, [1,0])[-1][0] > 0.99
assert feed_forward(network, [1,1])[-1][0] < 0.01

# fizz buzz task
def fizz_buzz_encode(x: int) -> list:
    if x % 15 == 0:
        return [0,0,0,1]
    elif x % 5 == 0:
        return [0,0,1,0]
    elif x % 3 == 0:
        return [0,1,0,0,]
    else:
        return [1,0,0,0]
assert fizz_buzz_encode(2) == [1,0,0,0]
assert fizz_buzz_encode(6) == [0,1,0,0]
assert fizz_buzz_encode(10) == [0,0,1,0]
assert fizz_buzz_encode(30) == [0,0,0,1]

def binary_encode(x: int) -> list:
    binary = []
    for i in range(10):
        binary.append(x % 2)
        x = x // 2
    return binary
#                             1 2 3 4 5 6 7 8 9 10
assert binary_encode(0) ==   [0,0,0,0,0,0,0,0,0,0]
assert binary_encode(1) ==   [1,0,0,0,0,0,0,0,0,0]
assert binary_encode(10) ==  [0,1,0,1,0,0,0,0,0,0]
assert binary_encode(101) == [1,0,1,0,0,1,1,0,0,0]
assert binary_encode(999) == [1,1,1,0,0,1,1,1,1,1]

xs = [binary_encode(n) for n in range(101,1024)]
ys = [fizz_buzz_encode(n) for n in range(101,1024)]

NUM_HIDDEN = 25

network = [
    [[random.random() for _ in range(10+1)] for _ in range(NUM_HIDDEN)],
    [[random.random() for _ in range(NUM_HIDDEN+1)] for _ in range(4)]
]
def sum_of_squares(l: list) -> float:
    return dot(l,l)

def substract(l1: list,l2: list) -> list:
    return [x - y for x,y in zip(l1,l2)]

def squared_distance(l1: list, l2: list) -> float:
    return sum_of_squares(substract(l1,l2))

learning_rate = 1.0
with tqdm.trange(500) as t:
    for epoch in t:
        epoch_loss = 0
        for x,y in zip(xs,ys):
            predicted = feed_forward(network, x)[-1]
            epoch_loss += squared_distance(predicted,y)
            gradients = sqerror_gradients(network,x,y)
            network = [[gradient_step(neuron, grad, -learning_rate) for neuron, grad in zip(layer, layer_grad)] for layer, layer_grad in zip(network,gradients)]
        t.set_description(f'fizz buzz (loss: {epoch_loss:.2f})')

def argmax(xs: list) -> int:
    return max(range(len(xs)), key = lambda i: xs[i])

assert argmax([0,-1]) == 0 # xs[0] == 0
assert argmax([-1,0]) == 1 # xs[1] == 0
assert argmax([-1,0,10,5,6,7,20]) == 6 # xs[6] == 20
corrects = 0
for i in range(1,101):
    x = binary_encode(i)
    predicted = argmax(feed_forward(network,x)[-1])
    actual = argmax(fizz_buzz_encode(i))
    labels = [str(i),'fizz','buzz','fizzbuzz']
    print(i, labels[predicted], labels[actual])
    if predicted == actual:
        corrects += 1
print(f'{corrects} / 100')
