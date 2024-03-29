import tensorflow as tf
import numpy as np
import os
import re


class Input:
    def __init__(self, label_path, img_path, height, width, channels, batch_size, save_path, label_rule: list,
                 criterion, captcha_len, decoding='utf8'):
        self.label_path = [label_path]
        self.img_path = [os.path.join(img_path, str(x) + '.jpeg') for x in range(10)]
        self.height = height
        self.width = width
        self.channels = channels
        self.batch_size = batch_size
        self.save_path = save_path
        self.label_rule = label_rule
        self.decoding = decoding
        self.criterion = criterion
        self.captcha_len = captcha_len

    def label_to_num(self, labels):
        """
        转换为one-hot编码
        :param labels: [[b"..."], [b"..."]]
        :return: None
        """
        # 2D=>1D
        labels = np.array(labels).flatten()
        # bytes=>string
        labels = list(map(lambda x: x.decode(self.decoding), labels))
        # one_hot = np.zeros([self.batch_size, self.captcha_len, len(self.label_rule)])
        # for index, label in enumerate(labels):
        #     for w, item in enumerate(re.match(self.criterion, label).groups()):
        #         one_hot[index, w, self.label_rule.index(item)] = 1
        # num_batch = tf.convert_to_tensor(one_hot)
        array = np.zeros([self.batch_size, self.captcha_len])
        for index, label in enumerate(labels):
            for i, v in enumerate(re.match(self.criterion, label).groups()):
                array[index, i] = self.label_rule.index(v)
        num_batch = tf.convert_to_tensor(array, dtype=tf.uint8)

        return num_batch

    def get_label_batch(self):
        # 创建文件队列
        label_q = tf.train.string_input_producer(tf.convert_to_tensor(self.label_path), shuffle=False)
        # 读取label
        reader = tf.TextLineReader()
        key, value = reader.read(label_q)
        # 解码
        records = [['None']]
        label = tf.decode_csv(value, record_defaults=records)
        # 批处理
        label_batch = tf.train.batch([label], batch_size=self.batch_size, num_threads=1, capacity=self.batch_size)
        print(label_batch)

        return label_batch

    def get_img_batch(self):
        # 创建文件路径队列
        img_q = tf.train.string_input_producer(tf.convert_to_tensor(self.img_path), shuffle=False)
        # 读取图片
        reader = tf.WholeFileReader()
        key, value = reader.read(img_q)
        # 解码图片
        value = tf.image.convert_image_dtype(tf.image.decode_jpeg(value, channels=self.channels), tf.float32)
        value_reshaped = tf.image.resize_images(value, [self.height, self.width])
        # 批处理
        img_batch = tf.train.batch([value_reshaped], batch_size=self.batch_size)
        print(img_batch)

        return img_batch

    def write_to_file(self, label_batch, img_batch):
        # 创建存储器
        writer = tf.python_io.TFRecordWriter(self.save_path)
        # 循环创建example，一次写入一个样本至文件
        for i in range(self.batch_size):
            label = label_batch[i].eval().tostring()  # int=>string
            img = img_batch[i].eval().tostring()  # float=>string
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
                "img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img]))
            }))
            writer.write(example.SerializeToString())
            print("--%s--写入成功!" % i)
        writer.close()

    def run(self):
        label_batch = self.get_label_batch()
        img_batch = self.get_img_batch()

        with tf.Session() as sess:
            # 创建线程协调器
            coord = tf.train.Coordinator()
            # 启动填充队列线程
            thread = tf.train.start_queue_runners(sess=sess, coord=coord)

            labels = sess.run(label_batch)
            print(labels[0])
            labels_str = self.label_to_num(labels)
            print(labels_str.eval())
            # 写入文件
            print('正在写入文件...')
            self.write_to_file(labels_str, img_batch)

            coord.request_stop()
            coord.join(thread)


if __name__ == '__main__':
    rule = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    i = Input('./test/text', './test/img', 30, 100, 3, 1000, './test/cifar.tfrecords', rule, r"(\w{1})(\w{1})(\w{1})(\w{1})", 4)
    i.run()
    print('done!')
