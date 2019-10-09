import tensorflow as tf


def compute_error(w, b, point):
    """
    计算损失值 (1/N)∑(wx+b-y)²
    :param w:
    :param b:
    :param point:[[x1,y1], [x2, y2]...]
    :return:
    """
    N = len(point)
    loss = 0

    for i in range(0, N):
        x = point[i, 0]
        y = point[i, 1]

        loss += (w * x + b - y)**2

    return loss / N


def gradient_descend(w, b, point, learning_rate):
    """
    梯度下降 w=w0-lr*((1/N)∑2x(w0x+b0-y))  b=b0-lr((1/N)∑2(wx+b0-y))
    :param w:
    :param b:
    :param point:[[x1,y1], [x2, y2]...]
    :param learning_rate:
    :return:
    """
    w_gradient = 0
    b_gradient = 0
    N = len(point)

    for i in range(0, N):
        x = point[i, 0]
        y = point[i, 1]

        w_gradient += (2 * x * ((w * x + b) - y))
        b_gradient += (2 * ((w * x + b) - y))
    w_gradient = (1 / N) * w_gradient
    b_gradient = (1 / N) * b_gradient

    w = w - learning_rate * w_gradient
    b = b - learning_rate * b_gradient

    return w, b


def gradient_descend_run(w, b, point, learning_rate, iter_steps):
    for i in range(iter_steps):
        w, b = gradient_descend(w, b, point, learning_rate)
        yield w, b


def run(point, learning_rate, iter_steps):
    w_init = 0
    b_init = 0
    count = 0

    print("初始化w:{}，b:{}".format(w_init, b_init))
    for w, b in gradient_descend_run(w_init, b_init, point, learning_rate, iter_steps):
        count += 1
        print("第{}次训练，w:{}, b:{}, loss:{}".format(
            count,
            w,
            b,
            compute_error(w, b, point)
        ))


def minst_train():
    (x, y), _ = tf.keras.datasets.mnist.load_data()
    print(x, y)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.int32)

    train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
    lr = 0.001
    w1 = tf.Variable(tf.random.truncated_normal(shape=[28 * 28, 256], stddev=0.01))
    b1 = tf.Variable(tf.zeros(shape=[256]))
    w2 = tf.Variable(tf.random.truncated_normal(shape=[256, 128], stddev=0.01))
    b2 = tf.Variable(tf.zeros(shape=[128]))
    w3 = tf.Variable(tf.random.truncated_normal(shape=[128, 10], stddev=0.01))
    b3 = tf.Variable(tf.zeros(shape=[10]))
    for i in range(20):
        for step, (x, y) in enumerate(train_db):
            x = tf.reshape(x, shape=[-1, 28 * 28])
            y = tf.one_hot(y, depth=10)

            with tf.GradientTape() as tap:
                y1 = x@w1 + b1  # [128, 256]
                y1 = tf.nn.relu(y1)
                y2 = y1@w2 + b2  # [128, 128]
                y2 = tf.nn.relu(y2)
                out = y2@w3 + b3  # [128, 10]

                loss = tf.reduce_mean(tf.square(y - out))
                gradients = tap.gradient(loss, [w1, b1, w2, b2, w3, b3])
                w1.assign_sub(lr * gradients[0])
                b1.assign_sub(lr * gradients[1])
                w2.assign_sub(lr * gradients[2])
                b2.assign_sub(lr * gradients[3])
                w3.assign_sub(lr * gradients[4])
                b3.assign_sub(lr * gradients[5])

                if step % 100 == 0:
                    print("i:", i, "step:", step, "loss:", float(loss))


if __name__ == '__main__':
    # minst_train()
    import numpy as np

    print(np.array([[1], [2], [3]]).flatten())
