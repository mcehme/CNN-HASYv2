import tensorflow as tf

class DataHandler():
    def get_data():
        pass

class MINSTDataHandler(DataHandler):
    def get_data():
        mnist = tf.keras.datasets.mnist
        return mnist.load_data()
