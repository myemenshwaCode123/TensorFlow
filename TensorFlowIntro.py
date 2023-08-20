import tensorflow as tf

# Starting off easy, with constants and operations in TensorFlow
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0, dtype=tf.float32)

print("node1: ", node1.numpy(), " node2: ", node2.numpy())

# Operations using constants
a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)

d = tf.multiply(a,b)
e = tf.add(c,b)
f = tf.subtract(d,e)

print("f: ", f.numpy())

# Covering tensor addition

# Create constant tensors for input values
g = tf.constant([1, 3], dtype=tf.float32)
h = tf.constant([2, 4], dtype=tf.float32)

# Perform element-wise addition on tensors a and b
adder_node = g + h

# Print the values of the resulting tensor using .numpy() to retrieve the values of the resulting tensor
# In TensorFlow version 2, tensors are by default eagerly executed,
# so you can directly access their values using the .numpy() method. (In version 1, it was "placeholder")
print("adder node: ", adder_node.numpy())

# To make the model trainable, we need to be able to modify the graph to get new outputs with the same input.
# Variables allows us to add trainable parameters to a graph

# Linear Model using Variables

# Define variables i and j with initial values
W = tf.Variable([0.3], dtype=tf.float32)
bias = tf.Variable([-0.3], dtype=tf.float32)

# Define the input data using a constant tensor
x = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)

# Define the linear model using TensorFlow operations
# Linear model: y = W * x + bias
linear_model = W * x + bias

# No need to initialize global variables in TensorFlow version 2

# Print the result of the linear model using the .numpy() method
print("linear model: ", linear_model.numpy())

# Define a placeholder for target values
# These target values are used for loss calculation
y = tf.constant([0, -1, -2, -3], dtype=tf.float32)

# Calculate squared differences and loss
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

# Print the loss using the .numpy() method
print("loss: ", loss.numpy())

# When our loss was outputted it was very bad, thus we must reduce the loss.
# TenserFlow provides optimizers that slowly change each variable in order to minimize the loss function

# Create a gradient descent optimizer
optimizer = tf.optimizers.SGD(learning_rate=0.01)

# Define the training step
def train_step():
    with tf.GradientTape() as tape:
        predictions = W * x + bias
        loss_value = tf.reduce_sum(tf.square(predictions - y))
    gradients = tape.gradient(loss_value, [W, bias])
    optimizer.apply_gradients(zip(gradients, [W, bias]))

# No need to initialize global variables in TensorFlow version 2

# Training loop
for _ in range(1000):
    train_step()

# Print the final values of W and b
print("Optimized W:", W.numpy())
print("Optimized bias:", bias.numpy())














