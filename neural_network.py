import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
input > weight > hidden layer 1 (activation function)
> weights > hidden layer 2 (activation function)
> weights > output layer

compare output to intended output > cost function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer, gradient descent, etc...)

backpropogation

feed forward + backpropogation = 1 epoch (one cycle of the network) 
'''


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
batch_size = 100

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10 # Example: [0,0,1,0,0,0,0,0,0,0] == 2

# images are 28*28, which we squash into a 
# one dimensional array of length 784
tensor_length = 28*28

# height x width
x = tf.placeholder('float', [None,tensor_length])
y = tf.placeholder('float')

def neural_network_model(data):

	# (input_data * weights) + biases

	# biases make sure that inputs of zero still
	# produce a non-zero output

	hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([tensor_length, n_nodes_hl1])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_layer_3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
					'biases':tf.Variable(tf.random_normal([n_classes]))}

	# (input_data * weights) + biases

	layer_1 = tf.add(tf.matmul(data,hidden_layer_1['weights']),hidden_layer_1['biases'])
	layer_1 = tf.nn.relu(layer_1)

	layer_2 = tf.add(tf.matmul(layer_1,hidden_layer_2['weights']),hidden_layer_2['biases'])
	layer_2 = tf.nn.relu(layer_2)

	layer_3 = tf.add(tf.matmul(layer_2,hidden_layer_3['weights']),hidden_layer_3['biases'])
	layer_3 = tf.nn.relu(layer_3)

	output = tf.matmul(layer_3, output_layer['weights']) + output_layer['biases']

	return output

def train_neural_network(x):

	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction, y) )

	optimizer = tf.train.AdamOptimizer().minimize(cost)

	num_of_epochs = 15

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(num_of_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x,epoch_y = mnist.train.next_batch(batch_size)
				_,c = sess.run([optimizer,cost], feed_dict={x:epoch_x, y:epoch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', num_of_epochs, 'loss', epoch_loss)
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('Accuracy',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)