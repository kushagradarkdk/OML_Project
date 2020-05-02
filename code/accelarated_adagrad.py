from keras.optimizers import Optimizer
from keras.legacy import interfaces
from keras import backend as K
import tensorflow as tf

class AAdagrad(Optimizer):
    """AAdagrad optimizer.
        Accelerated Method for Adagrad Optimizer.
    """
    def __init__(self, learning_rate=0.01, epsilon=None, decay=0., **kwargs):
        super(AAdagrad, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.decay = K.variable(decay, name='decay')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay


    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(AAdagrad, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        accumulators = [K.zeros(shape) for shape in shapes]
        past_gradientradients = [K.zeros(shape) for shape in shapes]
        self.weights = accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        learning_rate = self.learning_rate
        if self.initial_decay > 0:
            learning_rate = learning_rate * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        for p, g, a, past_gradient in zip(params, grads, accumulators, past_gradientradients):
            new_accumulator = a + K.square(g)  # update accumulator
            self.updates.append(K.update(a, new_accumulator))
            # the acceleration step based on the condition provided in the paper
            new_p = tf.where(past_gradient*g>0,p - learning_rate * (g + past_gradient) / (K.sqrt(new_accumulator) + self.epsilon),p - learning_rate * g / (K.sqrt(new_accumulator) + self.epsilon))
            # Applying constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            new_gradient = g
            self.updates.append(K.update(p, new_p))
            self.updates.append(K.update(past_gradient, new_gradient))
        return self.updates

