from keras.optimizers import Optimizer
from keras.legacy import interfaces
from keras import backend as K
import tensorflow as tf

class ASGD (Optimizer):
    # ASGD : Accelerated Stochastic Gradient Descent.
    def __init__(self, learning_rate=0.01, momentum=0., decay=0.,nesterov=False, **kwargs):
        super(ASGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(ASGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        gradients = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        learning_rate = self.learning_rate
        if self.initial_decay > 0:
            learning_rate = learning_rate * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        previous_gradients = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m , past_g in zip(params, gradients, moments, previous_gradients):
            #Important step. Note that in this step if the past gradient and gradient are pointing in the same direction and is above a certain
            #threshold then add the past_gradient with the current gradient and accelerate this step otherwise move as it was a normal SGD.
            v = tf.where(past_g*g>0,self.momentum * m - learning_rate *  (g + past_g),self.momentum * m - learning_rate * g)  # velocity
            self.updates.append(K.update(m, v))
            self.updates.append(K.update(past_g, g))
            if self.nesterov:
                new_p = p + self.momentum * v - learning_rate * g
            else:
                new_p = p + v
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates


