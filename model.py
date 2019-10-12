import tensorflow as tf
import numpy as np


class Model:
    def __init__(self, height, width, channels, batch_size, captcha_len, label_rule):
        self.height = height
        self.width = width
        self.channels = channels
        self.batch_size = batch_size
        self.captcha_len = captcha_len
        self.label_rule = label_rule
        # self.X = tf.placeholder(tf.float32, [None, self.height * self.width * self.channels])
        # self.Y = tf.placeholder(tf.uint8, [None, self.captcha_len * len(rule)])

    @staticmethod
    def weight_var(shape):
        w = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
        return w

    @staticmethod
    def bias_var(shape):
        b = tf.zeros(shape=shape)
        return b

    def predict(self, img_batch, label_batch):
        x = tf.reshape(img_batch, [self.batch_size, self.height, self.width, self.channels])
        y = tf.reshape(label_batch, [self.batch_size, self.captcha_len * len(rule)])

        with tf.variable_scope("cv1"):
            w1 = self.weight_var([3, 3, self.channels, 32])
            b1 = self.bias_var([32])

            x_relu1 = tf.nn.relu(tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME') + b1)
            x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.variable_scope("cv2"):
            w2 = self.weight_var([3, 3, 32, 64])
            b2 = self.bias_var([64])

            x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2)
            x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.variable_scope("cv3"):
            w3 = self.weight_var([3, 3, 64, 128])
            b3 = self.bias_var([128])

            x_relu3 = tf.nn.relu(tf.nn.conv2d(x_pool2, w3, strides=[1, 1, 1, 1], padding='SAME') + b3)
            output = tf.nn.max_pool(x_relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            print()

        # with tf.variable_scope("fc"):
        #     w1 = self.weight_var([output])
        #     b1 = tf.Variable(tf.zeros([1024]))
        #     y1 = tf.matmul(output, w1) + b1
        #     y1 = tf.nn.relu(y1)  # 降低线性关系
        #     y1 = tf.nn.dropout(y1, 1)
        #     # w2 = tf.Variable(tf.random_normal([256, 128]))
        #     # b2 = tf.Variable(tf.zeros([128]))
        #     # y2 = tf.matmul(y1, w2) + b2
        #     # y2 = tf.nn.relu(y2)
        #     # y2 = tf.nn.dropout(y2, 0.75)
        #     w3 = tf.Variable(tf.random_normal([1024, 10]))
        #     b3 = tf.Variable(tf.zeros(10))
        #     y_predict = tf.matmul(y1, w3) + b3


class CNN:
    def __init__(self, tfrecords_path: list, height, width, label_rule: list, channels, captcha_len, batch_size, steps):
        self.tfrecords_path = tfrecords_path
        self.height = height
        self.width = width
        self.label_rule = label_rule
        self.channels = channels
        self.captcha_len = captcha_len
        self.batch_size = batch_size
        self.steps = steps
        # self.X = tf.placeholder(tf.float32, [None, self.height * self.width * self.channels])
        # self.Y = tf.placeholder(tf.uint8, [None, len(self.label_rule)])

    def get_batch(self):
        # 构造文件队列
        q = tf.train.string_input_producer(self.tfrecords_path)
        # 创建阅读器
        reader = tf.TFRecordReader()
        key, value = reader.read(q)
        # 解析
        feature = tf.parse_single_example(value, features={
            "label": tf.FixedLenFeature([], tf.string),
            "img": tf.FixedLenFeature([], tf.string)
        })
        label = tf.decode_raw(feature["label"], tf.uint8)
        img = tf.decode_raw(feature["img"], tf.float32)
        label_reshape = tf.reshape(label, [self.captcha_len])
        img_reshape = tf.reshape(img, [self.height, self.width, self.channels])
        # 批处理
        label_batch, img_batch = tf.train.batch([label_reshape, img_reshape], batch_size=self.batch_size, num_threads=1,
                                                capacity=self.batch_size)

        return label_batch, img_batch

    def convert_to_one_hot(self, label):
        label_onehot = tf.one_hot(label, depth=len(self.label_rule), on_value=1.0, axis=2)

        return label_onehot

    def convert_to_one_hot1(self, labels):
        one_hot = np.zeros([self.batch_size, self.captcha_len, len(self.label_rule)])
        for index, label in enumerate(labels):
            for i, item in enumerate(label):
                one_hot[index, i, self.label_rule.index(item)] = 1

        return one_hot

    @staticmethod
    def weight_var(shape):
        w = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
        return w

    @staticmethod
    def bias_var(shape):
        b = tf.Variable(tf.zeros(shape=shape))
        return b

    def predict(self, img_batch):
        x = tf.reshape(img_batch, [self.batch_size, self.height, self.width, self.channels])

        with tf.variable_scope("cv1"):
            w1 = self.weight_var([3, 3, self.channels, 32])
            b1 = self.bias_var([32])

            x_relu1 = tf.nn.relu(tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME') + b1)
            x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.variable_scope("cv2"):
            w2 = self.weight_var([3, 3, 32, 64])
            b2 = self.bias_var([64])

            x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2)
            x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.variable_scope("cv3"):
            w3 = self.weight_var([3, 3, 64, 128])
            b3 = self.bias_var([128])

            x_relu3 = tf.nn.relu(tf.nn.conv2d(x_pool2, w3, strides=[1, 1, 1, 1], padding='SAME') + b3)
            output = tf.nn.max_pool(x_relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            output = tf.reshape(output, [output.shape[0].value, output.shape[1].value * output.shape[2].value * output.shape[3].value])
            print(output)

        with tf.variable_scope("fc"):
            w4 = tf.Variable(tf.random_normal([output.shape[1].value, 1024]))
            b4 = tf.Variable(tf.zeros([1024]))
            y4 = tf.matmul(output, w4) + b4
            y4 = tf.nn.relu(y4)  # 降低线性关系
            # y5 = tf.nn.dropout(y4, 1)
            w5 = tf.Variable(tf.random_normal([1024, self.captcha_len * len(self.label_rule)]))
            b5 = tf.Variable(tf.zeros([self.captcha_len * len(self.label_rule)]))
            y_predict = tf.matmul(y4, w5) + b5

        return y_predict, w5, b5

    def run(self):
        label_batch, img_batch = self.get_batch()
        y_predict, w, b = self.predict(img_batch)
        y_true = self.convert_to_one_hot(label_batch)
        y_predict_rshape = tf.reshape(y_predict, [self.batch_size, self.captcha_len, len(self.label_rule)])

        with tf.variable_scope("loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.reshape(y_true, [self.batch_size, self.captcha_len*len(self.label_rule)]), logits=y_predict))
            train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        with tf.variable_scope("accuracy"):
            equal_list = tf.equal(tf.argmax(y_true, axis=2), tf.argmax(y_predict_rshape, axis=2))
            accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            coord = tf.train.Coordinator()
            thread = tf.train.start_queue_runners(sess, coord=coord)
            for i in range(self.steps):
                sess.run([train_op])
                if i % 20 == 0:
                    print('step:', i, 'accuracy:', accuracy.eval(), 'loss:', loss.eval(), 'w:', w.eval()[0][0], 'b:', b.eval()[0])

            coord.request_stop()
            coord.join(thread)


if __name__ == '__main__':
    # rule = [i for i in range(104)]
    # c = CNN(["./cifar.tfrecords"], 30, 120, rule, 3, 3, 64, 1000)
    # c.run()
    # rule = list("abcdefghjkmnpqrstuvwxy")
    # c = CNN(["./test/cifar.tfrecords"], 30, 120, rule, 3, 4, 100, 1000)
    # c.run()
    rule = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    c = CNN(["./test/cifar.tfrecords"], 30, 100, rule, 3, 4, 100, 1000)
    c.run()

