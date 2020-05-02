from keras.optimizers import Optimizer
from keras.legacy import interfaces
from keras import backend as K
import tensorflow as tf

class AAdam (Optimizer):
    #AADAM optimizer which implemets the aspect of acceleration in normal ADAM algorithm.
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, **kwargs):
        super(AAdam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        
    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(AAdam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        #previous gradient
        previous_gradients = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        gradients = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        learning_rate = self.learning_rate
        if self.initial_decay > 0:
            learning_rate = learning_rate * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        learning_rate_t = learning_rate * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros((1,1)) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats 

        for p, g, m, v, vhat, past_g in zip(params, gradients, ms, vs, vhats, previous_gradients):
            v_t =  (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            if self.amsgrad:
                # past_g or momentum_current
                momentum_current = tf.where(tf.logical_and(tf.abs(past_g - g)>0.001,past_g*g>0),(self.beta_1 * m) +  (1. - self.beta_1) *  (g + past_g),(self.beta_1 * m) + (1. - self.beta_1) * g) 
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - learning_rate_t * momentum_current / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                # past_g or momentum_current
                momentum_current = tf.where(tf.logical_and(tf.abs(past_g - g)>0.001,past_g*g>0),(self.beta_1 * m) +  (1. - self.beta_1) *  (g + past_g),(self.beta_1 * m) + (1. - self.beta_1) * g) 
                p_t = p - learning_rate_t * momentum_current / (K.sqrt(v_t) + self.epsilon)
            self.updates.append(K.update(v, v_t))
            self.updates.append(K.update(past_g, g))
            self.updates.append(K.update(m, momentum_current))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    
