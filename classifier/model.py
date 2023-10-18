import tensorflow as tf

class Model():
    def __init__(self):
        self.layers = list()
        self.compiled = False
    def add_layer(self, layer):
        if self.compiled:
            raise Exception("Cannot add layer to compiled model")
        self.layers.append(layer)
    def compile(self, optimizer, loss, metrics):
        if self.compiled:
            raise Exception("Cannot compile already compiled model")
        self.model = tf.keras.models.Sequential(self.layers)
        self.model.compile(optimizer, loss, metrics)
        self.compiled = True
    def fit(self, X_train, y_train, epochs):
        if not self.compiled:
            raise Exception("Model must be compiled before fitting")
        self.model.fit(X_train, y_train, epochs=epochs)
    def evaluate(self, X_test, y_test):
        if not self.compiled:
            raise Exception("Model must be compiled before evaluating")
        return self.model.evaluate(X_test, y_test)
    def store(self, dir):
        if not self.compiled:
            raise Exception("Model must be compiled before storing")
        self.model.save_weights(dir)
    
