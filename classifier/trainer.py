import tensorflow as tf
import copy, json, os

class Trainer():
    def __init__(self, num_classes, data, epochs):
        self.data = data
        self.epochs = epochs
        self.num_classes = num_classes
        
    
    def tune(self, file_location, stats_dump, dropouts, dense_activations,
              dense_units, cnn_activations, pool_sizes,
                kernel_sizes, filters, layers):
        self.best = 0
        self.file_location = file_location
        self.stats_dump = stats_dump
        self.__dropout_tune(dropouts, dense_activation_tune=dense_activations,
              units_tune=dense_units, cnn_activation_tune=cnn_activations, pool_tune=pool_sizes,
                kernel_tune=kernel_sizes, filter_tune=filters, layer_tune=layers)
        mod = tf.keras.models.load_model(self.file_location)
        mod.summary()
        print(self.best_config)

    def __dropout_tune(self, dropouts, **kwargs):
        dense_activation_tune = kwargs.get('dense_activation_tune', ('relu',))
        if 'dense_activation_tune' in kwargs:
            del kwargs['dense_activation_tune']
        dropout_kwargs = kwargs.get('dropout_kwargs', {})
        for dropout in dropouts:
            dropout_kwargs['rate'] = dropout
            kwargs['dropout_kwargs'] = dropout_kwargs
            self.__dense_activation_tune(dense_activation_tune, **kwargs)
    
    def __dense_activation_tune(self, activations, **kwargs):
        units_tune = kwargs.get('units_tune', (64, 128, 256))
        if 'units_tune' in kwargs:
            del kwargs['units_tune']
        dense_kwargs = kwargs.get('dense_kwargs', {})
        for activation in activations:
            dense_kwargs['activation'] = activation
            kwargs['dense_kwargs'] = dense_kwargs
            self.__units_tune(units_tune, **kwargs)

    def __units_tune(self, units, **kwargs):
        cnn_activation_tune = kwargs.get('cnn_activation_tune', ('relu',))
        if 'cnn_activation_tune' in kwargs:
            del kwargs['cnn_activation_tune']
        dense_kwargs = kwargs.get('dense_kwargs', {})
        for unit in units:
            dense_kwargs['units'] = unit
            kwargs['dense_kwargs'] = dense_kwargs
            self.__cnn_activation_tune(cnn_activation_tune, **kwargs)
    def __cnn_activation_tune(self, activations, **kwargs):
        pool_tune = kwargs.get('pool_tune', (2, 3))
        if 'pool_tune' in kwargs:
            del kwargs['pool_tune']
        conv_kwargs = kwargs.get('conv_kwargs', {})
        for activation in activations:
            conv_kwargs['activation'] = activation
            kwargs['conv_kwargs'] = conv_kwargs
            self.__pool_tune(pool_tune, **kwargs)

    def __pool_tune(self, pools, **kwargs):
        kernel_tune = kwargs.get('kernel_tune', (2, 3))
        if 'kernel_tune' in kwargs:
            del kwargs['kernel_tune']
        pool_kwargs = kwargs.get('pool_kwargs', {})
        for pool in pools:
            pool_kwargs['pool_size'] = pool
            kwargs['pool_kwargs'] = pool_kwargs
            self.__kernel_tune(kernel_tune, **kwargs)

    def __kernel_tune(self, kernels, **kwargs):
        filter_tune = kwargs.get('filter_tune', (16, 32))
        if 'filter_tune' in kwargs:
            del kwargs['filter_tune']
        conv_kwargs = kwargs.get('conv_kwargs', {})
        for kernel in kernels:
            conv_kwargs['kernel_size'] = kernel
            kwargs['conv_kwargs'] = conv_kwargs
            self.__filter_tune(filter_tune, **kwargs)


    def __filter_tune(self, filters, **kwargs):
        layer_tune = kwargs.get('layer_tune', (1, 3))
        if 'layer_tune' in kwargs:
            del kwargs['layer_tune']
        conv_kwargs = kwargs.get('conv_kwargs', {})
        for filter in filters:
            conv_kwargs['filters'] = filter
            kwargs['conv_kwargs'] = conv_kwargs
            self.__layer_tune(layer_tune, **kwargs)


    def __layer_tune(self, layers, **kwargs):
        conv_kwargs = kwargs.get('conv_kwargs', {})
        pool_kwargs = kwargs.get('pool_kwargs', {})
        flatten_kwargs = kwargs.get('flatten_kwargs', {})
        dense_kwargs = kwargs.get('dense_kwargs', {})
        dropout_kwargs = kwargs.get('dropout_kwargs', {})
        for i in layers:
            for j in layers:
                mod = tf.keras.models.Sequential()
                self.__add_layers(mod, i, self.__conv_layer, conv_kwargs, pool_kwargs)
                self.__add_layers(mod, 1, self.__flatten, flatten_kwargs)
                self.__add_layers(mod, j, self.__dense_layer, dense_kwargs, dropout_kwargs)
                copy_kwargs = dense_kwargs.copy()
                copy_kwargs['units'] = self.num_classes
                self.__add_layers(mod, 1, self.__dense_layer, copy_kwargs, {"rate":0})
                mod.compile('adam', tf.keras.losses.CategoricalCrossentropy(from_logits=True), ['accuracy'])
                mod.fit(*self.data[0], epochs=self.epochs, verbose=0)
                stats = mod.evaluate(*self.data[1])

                details = copy.deepcopy(kwargs)
                details['conv_layers'] = i
                details['dense_layers'] = j
                details['loss'] = stats[0]
                details['accuracy'] = stats[1]
                if os.path.exists(self.stats_dump):
                    with open(self.stats_dump, 'r') as f:
                        stats_dump = json.load(f)
                else:
                    stats_dump = []
                stats_dump.append(details)

                with open(self.stats_dump, 'w') as f:
                    json.dump(stats_dump, f)

                if stats[1] > self.best:
                    self.best = stats[1]
                    self.best_config = copy.deepcopy(kwargs)
                    mod.save(self.file_location)

    
    def __add_layers(self, mod, num_layers, callback, *args):
        for _ in range(0, num_layers):
            for layer in callback(*args):
                mod.add(layer)

    def __conv_layer(self, conv_kwargs, pool_kwargs):
        return tf.keras.layers.Conv2D(**conv_kwargs), tf.keras.layers.MaxPool2D(**pool_kwargs) 
    def __flatten(self, flatten_kwargs):
        return tf.keras.layers.Flatten(**flatten_kwargs),
    def __dense_layer(self, dense_kwargs, dropout_kwargs):
        return tf.keras.layers.Dense(**dense_kwargs), tf.keras.layers.Dropout(**dropout_kwargs)


