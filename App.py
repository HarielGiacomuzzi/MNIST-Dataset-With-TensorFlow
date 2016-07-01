import tensorflow as tf
import math
import numpy
import time
import os
from tensorflow.python.framework import dtypes
import collections

# Global Parameters
# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
BATCH_SIZE = 100
MAX_STEPS = 10000
TRAIN_DIR = 'data'

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

# Programm Methods

# this method loads the data, creates the graph in memory and executes the training phases
def run_training():
	# placeholder variables for loading the files later...
	train_images = None
	train_labels = None
	test_images  = None
	test_labels  = None
	# reads the images from the binary file into a 4D numpy uint8 array
	with open(os.path.abspath("train-images.idx3-ubyte"),"rb") as bytestream :
		print('Loading Images to Memory')
		magic = _read32(bytestream)
		if magic != 2051:
			raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, filename))
		num_images = _read32(bytestream)
		rows = _read32(bytestream)
		cols = _read32(bytestream)
		buf = bytestream.read(rows * cols * num_images)
		data = numpy.frombuffer(buf, dtype=numpy.uint8)
		data = data.reshape(num_images, rows, cols, 1)
		train_images = data

	# reads the labels to the memory

	with open(os.path.abspath("train-labels.idx1-ubyte"),'rb') as bytestream : 
		print('Loading labels from file')
		magic = _read32(bytestream)
		if magic != 2049:
			raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, filename))
		num_items = _read32(bytestream)
		buf = bytestream.read(num_items)
		labels = numpy.frombuffer(buf, dtype=numpy.uint8)
		train_labels = labels

	# reads the images from the binary file into a 4D numpy uint8 array

	with open(os.path.abspath("t10k-images.idx3-ubyte"),"rb") as bytestream :
		print('Loading Images to Memory')
		magic = _read32(bytestream)
		if magic != 2051:
			raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, filename))
		num_images = _read32(bytestream)
		rows = _read32(bytestream)
		cols = _read32(bytestream)
		buf = bytestream.read(rows * cols * num_images)
		data = numpy.frombuffer(buf, dtype=numpy.uint8)
		data = data.reshape(num_images, rows, cols, 1)
		test_images = data

	# reads the labels to the memory

	with open(os.path.abspath("t10k-labels.idx1-ubyte"),'rb') as bytestream : 
		print('Loading labels from file')
		magic = _read32(bytestream)
		if magic != 2049:
			raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, filename))
		num_items = _read32(bytestream)
		buf = bytestream.read(num_items)
		labels = numpy.frombuffer(buf, dtype=numpy.uint8)
		test_labels = labels

	# divides the set of 10K in the middle to validation and traind
	validation_images = train_images[:5000]
	validation_labels = train_labels[:5000]
	train_images = train_images[5000:]
	train_labels = train_labels[5000:]

	train = DataSet(train_images, train_labels, dtype=dtypes.float32)
	validation = DataSet(validation_images, validation_labels, dtype=dtypes.float32)
	test = DataSet(test_images, test_labels, dtype=dtypes.float32)
	data_sets = Datasets(train=train, validation=validation, test=test)

	with tf.Graph().as_default():
		# Generate placeholders for the images and labels.
		images_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_PIXELS) )
		labels_placeholder = tf.placeholder(tf.int32, shape=(BATCH_SIZE))

		# Build a Graph that computes predictions from the inference model.
		logits = inference(images_placeholder, 7, [784,2500,2000,1500,1000,500,10], ['hidden1','hidden2','hidden3','hidden4','hidden5','hidden6','softmax_linear'])

		# Add to the Graph the Ops for loss calculation.
		loss = Loss(logits, labels_placeholder)

		# Add to the Graph the Ops that calculate and apply gradients.
		train_op = training(loss, 0.01)

		# Add the Op to compare the logits to the labels during evaluation.
		eval_correct = evaluation(logits, labels_placeholder)

		# Build the summary operation based on the TF collection of Summaries.
		summary_op = tf.merge_all_summaries()

		# Add the variable initializer Op.
		init = tf.initialize_all_variables()

		# Create a saver for writing training checkpoints.
		saver = tf.train.Saver()

		# Create a session for running Ops on the Graph.
		sess = tf.Session()

		# Instantiate a SummaryWriter to output summaries and the Graph.
		summary_writer = tf.train.SummaryWriter(TRAIN_DIR, sess.graph)

		# And then after everything is built:
		# Run the Op to initialize the variables.
		sess.run(init)

		# Start the training loop.
		for step in xrange(MAX_STEPS):
			start_time = time.time()

			# Fill a feed dictionary with the actual set of images and labels
			# for this particular training step.
			feed_dict = fill_feed_dict( data_sets.train, images_placeholder, labels_placeholder )

			# Run one step of the model.  The return values are the activations
			# from the `train_op` (which is discarded) and the `loss` Op.  To
			# inspect the values of your Ops or variables, you may include them
			# in the list passed to sess.run() and the value tensors will be
			# returned in the tuple from the call.
			_, loss_value = sess.run([train_op, loss],feed_dict=feed_dict)
			duration = time.time() - start_time

			# Write the summaries and print an overview fairly often.
			if step % 100 == 0:
				# Print status to stdout.
				print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
				# Update the events file.
				summary_str = sess.run(summary_op, feed_dict=feed_dict)
				summary_writer.add_summary(summary_str, step)
				summary_writer.flush()


			# Save a checkpoint and evaluate the model periodically.
			if (step + 1) % 1000 == 0 or (step + 1) == MAX_STEPS:
				saver.save(sess, TRAIN_DIR, global_step=step)
				# Evaluate against the training set.
				print('Training Data Eval:')
				do_eval(sess,eval_correct,images_placeholder,labels_placeholder,data_sets.train)
				# Evaluate against the validation set.
				print('Validation Data Eval:')
				do_eval(sess,eval_correct,images_placeholder,labels_placeholder,data_sets.validation)
				# Evaluate against the test set.
				print('Test Data Eval:')
				do_eval(sess,eval_correct,images_placeholder,labels_placeholder,data_sets.test)

def do_eval(sess,eval_correct,images_placeholder,labels_placeholder,data_set):
	"""Runs one evaluation against the full epoch of data.
	Args:
	sess: The session in which the model has been trained.
	eval_correct: The Tensor that returns the number of correct predictions.
	images_placeholder: The images placeholder.
	labels_placeholder: The labels placeholder.
	data_set: The set of images and labels to evaluate, from
	input_data.read_data_sets().
	"""
	# And run one epoch of eval.
	true_count = 0.0  # Counts the number of correct predictions.
	steps_per_epoch = data_set.num_examples // BATCH_SIZE
	num_examples = steps_per_epoch * BATCH_SIZE
	for step in xrange(steps_per_epoch):
		feed_dict = fill_feed_dict(data_set,images_placeholder,labels_placeholder)
		true_count += sess.run(eval_correct, feed_dict=feed_dict)
	precision = true_count / num_examples
	print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %(num_examples, true_count, precision))

def fill_feed_dict(data_set, images_pl, labels_pl):
	"""Fills the feed_dict for training the given step.
	A feed_dict takes the form of:
	feed_dict = {
	<placeholder>: <tensor of values to be passed for placeholder>,
	....
	}
	Args:
	data_set: The set of images and labels, from input_data.read_data_sets()
	images_pl: The images placeholder, from placeholder_inputs().
	labels_pl: The labels placeholder, from placeholder_inputs().
	Returns:
	feed_dict: The feed dictionary mapping from placeholders to values.
	"""
	# Create the feed_dict for the placeholders filled with the next
	# `batch size ` examples.
	images_feed, labels_feed = data_set.next_batch( BATCH_SIZE, False )
	feed_dict = { images_pl: images_feed, labels_pl: labels_feed, }
	return feed_dict

# reads 32 bits from a binary file
def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

# This method creates the layers of the network
def inference(inputData, numberOfLayers, sizeOfLayer, nameOfLayer):
	"""Build the model up to where it may be used for inference.
  		Args:
    		input: Images placeholder, from inputs.  Type: I Don't know, put something you like :)
    		numberOfLayers: Number of hidden layers. Type: Integer
    		sizeOfLayer: Size of each hidden layer.  Type: List of Integer
    		nameOfLayer: Name of each layer          Type: List of Strings
  		Returns:
    		softmax_linear: Output tensor with the computed logits.
  	"""
  	layersDict = []
  	#  InputLayer
  	with tf.name_scope(nameOfLayer[0]):
		weights = tf.Variable( tf.truncated_normal([IMAGE_PIXELS, sizeOfLayer[0]], stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))), name='weights' )
		biases = tf.Variable( tf.zeros([sizeOfLayer[0]]), name='biases' )
		layersDict.append(tf.nn.relu(tf.matmul(inputData, weights) + biases))

  	#  Middle Layers
	for i in range(1,numberOfLayers-1):
		with tf.name_scope(nameOfLayer[i]):
				weights = tf.Variable( tf.truncated_normal([sizeOfLayer[i-1], sizeOfLayer[i]], stddev=1.0 / math.sqrt(float(sizeOfLayer[i-1]))), name='weights' )
				biases = tf.Variable( tf.zeros([sizeOfLayer[i]]), name='biases' )
				layersDict.append(tf.nn.relu(tf.matmul(layersDict[i-1], weights) + biases))

	# Output
  	with tf.name_scope(nameOfLayer[numberOfLayers-1]):
		weights = tf.Variable( tf.truncated_normal([sizeOfLayer[numberOfLayers-2], NUM_CLASSES], stddev=1.0 / math.sqrt(float(sizeOfLayer[numberOfLayers-2]))), name='weights')
		biases = tf.Variable(tf.zeros([NUM_CLASSES]),name='biases')
		layersDict.append(tf.matmul(layersDict[numberOfLayers-2], weights) + biases)

	return layersDict[numberOfLayers-1]


def Loss(logits, labels):
	"""Calculates the loss from the logits and the labels.
	Args:
	logits: Logits tensor, float - [batch_size, NUM_CLASSES].
	labels: Labels tensor, int32 - [batch_size].
	Returns:
	loss: Loss tensor of type float.
	"""
	labels = tf.to_int64(labels)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits( logits, labels, name='xentropy' )
	loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
	return loss

def training(loss, learning_rate):
	"""Sets up the training Ops.
	Creates a summarizer to track the loss over time in TensorBoard.
	Creates an optimizer and applies the gradients to all trainable variables.
	The Op returned by this function is what must be passed to the
	`sess.run()` call to cause the model to train.
	Args:
	loss: Loss tensor, from loss().
	learning_rate: The learning rate to use for gradient descent.
	Returns:
	train_op: The Op for training.
	"""
	# Add a scalar summary for the snapshot loss.
	tf.scalar_summary(loss.op.name, loss)
	# Create the gradient descent optimizer with the given learning rate.
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	# Create a variable to track the global step.
	global_step = tf.Variable(0, name='global_step', trainable=False)
	# Use the optimizer to apply the gradients that minimize the loss
	# (and also increment the global step counter) as a single training step.
	train_op = optimizer.minimize(loss, global_step=global_step)
	return train_op


def evaluation(logits, labels):
	"""Evaluate the quality of the logits at predicting the label.
	Args:
	logits: Logits tensor, float - [batch_size, NUM_CLASSES].
	labels: Labels tensor, int32 - [batch_size], with values in the
	range [0, NUM_CLASSES).
	Returns:
	A scalar int32 tensor with the number of examples (out of batch_size)
	that were predicted correctly.
	"""
	# For a classifier model, we can use the in_top_k Op.
	# It returns a bool tensor with shape [batch_size] that is true for
	# the examples where the label is in the top k (here k=1)
	# of all logits for that example.
	correct = tf.nn.in_top_k(logits, labels, 1)
	# Return the number of true entries.
	return tf.reduce_sum(tf.cast(correct, tf.int32))



# class dataset
class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      assert images.shape[3] == 1
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

# Main Program
if __name__ == '__main__':
	# executes the training steps
	run_training()


