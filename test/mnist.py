import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class Test:
    def __init__(self, steps, batch_size, lr):
        self.X = tf.placeholder(tf.float32, [None, 784])
        self.Y = tf.placeholder(tf.int8, [None, 10])
        self.steps = steps
        self.batch_size = batch_size
        self.lr = lr
        self.mnist = input_data.read_data_sets(r"D:\crawl_datasource\yzm\data\\", one_hot=True)

    @staticmethod
    def weight_var(shape):
        w = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
        return w

    @staticmethod
    def bias_var(shape):
        b = tf.zeros(shape=shape)
        return b

    def model(self):
        """
        [batch, 784] * [784, 256] + [256] = [batch, 256]
        [batch, 256] * [256, 128] + [128] = [batch, 128]
        [batch, 128] * [128, 10] + [10] = [batch, 10]
        :return:
        """
        x = tf.reshape(self.X, [-1, 28, 28, 1])
        with tf.variable_scope("cv1"):
            w1 = self.weight_var([3, 3, 1, 32])
            b1 = self.bias_var([32])

            x_relu1 = tf.nn.relu(tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME') + b1)  # [None, 28, 28, 1]=>[None, 28, 28, 32]
            x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # ksize池化窗口大小 [None, 14, 14, 32]

        with tf.variable_scope('cv2'):
            w2 = self.weight_var([3, 3, 32, 64])
            b2 = self.bias_var([64])

            x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2)
            x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # output = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # output = tf.reshape(output, [-1, 7 * 7 * 64])

        with tf.variable_scope('cv3'):
            w3 = self.weight_var([3, 3, 64, 128])
            b3 = self.bias_var([128])

            x_relu3 = tf.nn.relu(tf.nn.conv2d(x_pool2, w3, strides=[1, 1, 1, 1], padding='SAME') + b3)
            output = tf.nn.max_pool(x_relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            print(output)
            output = tf.reshape(output, [-1, 4 * 4 * 128])

        with tf.variable_scope("full_connection"):
            w1 = tf.Variable(tf.random_normal([4 * 4 * 128, 256]))
            b1 = tf.Variable(tf.zeros([256]))
            y1 = tf.matmul(output, w1) + b1
            y1 = tf.nn.relu(y1)  # 降低线性关系
            # w2 = tf.Variable(tf.random_normal([256, 128]))
            # b2 = tf.Variable(tf.zeros([128]))
            # y2 = tf.matmul(y1, w2) + b2
            # y2 = tf.nn.softmax(y2)
            w3 = tf.Variable(tf.random_normal([256, 10]))
            b3 = tf.Variable(tf.zeros(10))
            y_predict = tf.matmul(y1, w3) + b3

        with tf.variable_scope("compute_loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=y_predict))
            train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

        with tf.variable_scope("compute_accuracy"):
            equal_list = tf.equal(tf.argmax(self.Y, axis=1), tf.argmax(y_predict, axis=1))
            accuracy_op = tf.reduce_mean(tf.cast(equal_list, tf.float32))
        init_op = tf.global_variables_initializer()

        return train_op, accuracy_op, init_op

    def run(self):
        model, accuracy, init = self.model()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(self.steps):
                train_x, train_y = self.mnist.train.next_batch(self.batch_size)
                sess.run(model, feed_dict={self.X: train_x, self.Y: train_y})
                if i % 20 == 0:
                    print("第%s次训练，准确率为：%s" % (
                        i,
                        sess.run(accuracy, feed_dict={self.X: train_x, self.Y: train_y})
                    ))


if __name__ == '__main__':
    fc = Test(6001, 128, 0.01)
    fc.run()
