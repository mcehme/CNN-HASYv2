from classifier import model, datahandler
import tensorflow as tf 
class Trainer():
    def __init__(self):
        self.data = datahandler.MINSTDataHandler().get_data()

    def batch_train(self, low, high):
        mod = model.Model()



    def __add_layers(self, num_layers, callback, **kwargs):
        for _ in range(0, num_layers):
            callback(**kwargs)


    def __conv_layer(self, filters, kernel_size, activation, pool_size):
        return tf.keras.layers.Conv2D(filters,kernel_size, activation), tf.keras.layers.MaxPool2D(pool_size) 
    def __flatten(self):
        return tf.keras.layers.Flatten()
    def __dense_layer(self, units, activation, dropout):
        return tf.keras.layers.Dense(units, activation), tf.keras.layers.Dropout(dropout)


