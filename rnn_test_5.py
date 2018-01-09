import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

# max_len = 150
# num = 10664
class ReactionData():
    def __init__(self, st ='training'):
        if st == 'training':
            self.data = np.load('training_set_r.npy').tolist()
            self.labels = np.load('training_set_label.npy').tolist()
            print('Training set ready! Length: %d'%len(self.labels))
        elif st == 'test':
            self.data = np.load('test_set_r.npy').tolist()
            self.labels = np.load('test_set_label.npy').tolist()
            data_cl = [[], [], [], [], [], [], []]
            labels_cl = [[], [], [], [], [], [], []]
            for i in range(len(self.labels)):
                for j in range(7):
                    if self.labels[i][j] == 1.0:
                        data_cl[j].append(self.data[i])
                        labels_cl[j].append(self.labels[i])
                        break
            for k in range(7):
                print(len(data_cl[k]))
            self.data_cl = data_cl
            self.labels_cl = labels_cl
            print('Test set ready! Length: %d' % len(self.labels))
        elif st == 'validation':
            self.data = np.load('validation_set_r.npy').tolist()
            self.labels = np.load('validation_set_label.npy').tolist()
        self.batch_id = 0

    def next(self, batch_size):
        # Return a batch of data. When dataset end is reached, start over.
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        # 数据集已经按长度从小到大排序，因此都按照最长的那个
        # 一个batch里按照最长的长度补齐0.0
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels

learning_rate = 0.01
training_steps = 5000
batch_size = 64
display_step = 200
# Network Parameters
# 神经网络参数
num_input = 1 # MNIST data input (img shape: 28*28) 输入层
timesteps = 150 # timesteps 28 长度
num_hidden = 128 # hidden layer num of features 隐藏层神经元数
num_classes = 7 # MNIST total classes (0-9 digits) 输出数量，分类类别 0~9

trainset = ReactionData('training')
testset = ReactionData('test')
valset = ReactionData('validation')

# tf Graph input
# 输入数据占位符
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])
# Define weights
# 定义权重
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}
# 定义RNN模型
def RNN(x, weights, biases):
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    # 输入x转换成(128 batch * 28 steps, 28 inputs)
    x = tf.unstack(x, timesteps, 1)
    # Define a lstm cell with tensorflow
    # 基本LSTM循环网络单元 BasicLSTMCell
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']
logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)
# Define loss and optimizer
# 定义损失函数
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
# 定义优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
# Start training

saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    for step in range(1, training_steps+1):
        batch_x, batch_y = trainset.next(batch_size)
        # Reshape data to get 28 seq of 28 elements
        # batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " +
                  "{:.4f}".format(loss) + ", Training Accuracy= " +
                  "{:.3f}".format(acc))
    print("Optimization Finished!")
    save_path = saver.save(sess, "model_0.ckpt")
    # Calculate accuracy for 128 mnist test images
    test_data = testset.data
    test_label = testset.labels
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

    for l in range(7):
        print(l)
        test_data = testset.data_cl[l]
        test_label = testset.labels_cl[l]
        print('Pred:', sess.run(prediction, feed_dict={X: test_data}))
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

