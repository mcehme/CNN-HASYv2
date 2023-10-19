import tensorflow as tf

class DataHandler():
    def get_data():
        pass

class MINSTDataHandler(DataHandler):
    def get_data():
        mnist = tf.keras.datasets.mnist
        data = mnist.load_data()
        X1, y1, X2, y2 = *data[0], *data[1]
        X1 = X1.reshape(-1, 28, 28, 1).astype('float32')/255
        X2 = X2.reshape(-1, 28, 28, 1).astype('float32')/255
        y1 = tf.keras.utils.to_categorical(y1)
        y2 = tf.keras.utils.to_categorical(y2)
        return (X1, y1), (X2, y2)
