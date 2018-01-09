import numpy as np
import tensorflow as tf


# max_len = 150
# num = 10664
class ReactionData():
    def __init__(self, st ='training'):
        if st == 'training':
            self.data = np.load('training_set_2.npy').tolist()
            self.labels = np.load('training_set_label_2.npy').tolist()
            print('Training set ready! Length: %d'%len(self.labels))
        elif st == 'test':
            self.data = np.load('test_set_2.npy').tolist()
            self.labels = np.load('test_set_label_2.npy').tolist()
            print('Test set ready! Length: %d' % len(self.labels))
        elif st == 'validation':
            self.data = np.load('validation_set_2.npy').tolist()
            self.labels = np.load('validation_set_label_2.npy').tolist()
        self.batch_id = 0

    def next(self, batch_size):
        # Return a batch of data. When dataset end is reached, start over.
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_seqlen = ([50] * len(batch_labels))
        # 数据集已经按长度从小到大排序，因此都按照最长的那个
        # 一个batch里按照最长的长度补齐0.0
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, [50] * len(batch_seqlen)

# 超参数
learning_rate = 0.01
training_steps = 5000
batch_size = 64
display_step = 200

# 网络参数
seq_max_len = 50 # Sequence max length
n_hidden = 128 # hidden layer num of features
n_classes = 7 # linear sequence or not

trainset = ReactionData('training')
testset = ReactionData('test')
valset = ReactionData('validation')

# tf Graph input
x = tf.placeholder("float", [None, seq_max_len, 1])
y = tf.placeholder("float", [None, n_classes])
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
biases = {'out': tf.Variable(tf.random_normal([n_classes]))}


def dynamicRNN(x, seqlen, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, seq_max_len, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']

pred = dynamicRNN(x, seqlen, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    for step in range(1, training_steps+1):
        batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
        if step % display_step == 0 or step == 1:
            # Calculate batch accuracy & loss
            acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))

            val_seqlen, val_label, val_data = [50] * len(valset.labels), valset.labels, valset.data
            print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: val_data, y: val_label, seqlen: val_seqlen}))

    print("Optimization Finished!")

    # Calculate accuracy
    test_seqlen, test_label, test_data = [50] * len(testset.labels), testset.labels, testset.data
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label, seqlen: test_seqlen}))