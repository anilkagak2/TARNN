import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell_impl import RNNCell

class TARNNCell(RNNCell):
    def __init__(self, hidden_size, n_features, n_timesteps, alpha_val=0.01, sigma='relu', K=2, gating=True):
        print("TARNNCell -> ", ' K=', K)
        super(TARNNCell, self).__init__()
        assert( n_features <= hidden_size )
        self.hidden_size = hidden_size
        self.sigma = sigma
        self.t = 0
        self.n_timesteps = n_timesteps
        self.n_features  = n_features
        self.K = K
        self.gating = gating
        self.alpha_val = alpha_val
        self.n_hidden = hidden_size 

    @property
    def state_size(self):
        return 2 * self.hidden_size

    @property
    def output_size(self):
        return self.hidden_size

    def NL(self, x, sigma='relu'):
        if sigma == 'relu': return tf.nn.relu(x)
        elif sigma == 'tanh': return tf.nn.tanh(x)
        elif sigma == 'sigmoid': return tf.nn.sigmoid(x)
        raise Exception('Non-linearity not found..')

    def call(self, x, h):
        #print('t = ', self.t, ' , n_timesteps = ', self.n_timesteps)
        alpha_val = self.alpha_val
        #initializer = tf.contrib.layers.xavier_initializer()
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32)
        bias_update_init = init_ops.constant_initializer(0.0, dtype=tf.float32)
        
        hidden_size = self.n_hidden #self.hidden_size
        n_features  = x.get_shape()[-1]  #self.n_features
        
        with vs.variable_scope("B_TARNNCell"):
            b1 = vs.get_variable("b1", [hidden_size], initializer=bias_update_init)
            b2 = vs.get_variable("b2", [hidden_size], initializer=bias_update_init)

            B1 = vs.get_variable("B1", [n_features, hidden_size], initializer=initializer)
            B2 = vs.get_variable("B2", [hidden_size, hidden_size], initializer=initializer)
            
            W1 = vs.get_variable("W1", [n_features, hidden_size], initializer=initializer)
            W2 = vs.get_variable("W2", [n_features, hidden_size], initializer=initializer)

            U1 = vs.get_variable("U1", [hidden_size, hidden_size], initializer=initializer)
            U2 = vs.get_variable("U2", [hidden_size, hidden_size], initializer=initializer)      

            Wg = vs.get_variable("Wg", [n_features, hidden_size], initializer=initializer)
            Ug = vs.get_variable("Ug", [hidden_size, hidden_size], initializer=initializer)        
            #bg = vs.get_variable("bg", [hidden_size], initializer=init_ops.constant_initializer(-3.0, dtype=tf.float32))
            bg = vs.get_variable("bg", [hidden_size], initializer=init_ops.constant_initializer(0.0, dtype=tf.float32))

            alpha_init = 0.01
            if alpha_val is not None: alpha_init = alpha_val
            beta_init = 1.0 - alpha_init
            #print('alpha_init = ', alpha_init)
            #print('beta_init = ', beta_init)
            #A1 = tf.Variable(alpha_init * tf.ones((self.K, self.n_timesteps)), dtype=tf.float32, name='alpha')
            
            A1 = vs.get_variable("alpha", [self.K, self.n_timesteps], 
                                 initializer=init_ops.constant_initializer(alpha_init, dtype=tf.float32))
        
        REM = self.hidden_size - self.n_features
        B = tf.shape(x)[0]
        
        #print('\n\n')
        h, prev_x = tf.split(value=h, num_or_size_splits=2, axis=1)
        xtp, _ = tf.split(prev_x, [self.n_features, REM], axis=1)
        
        #print('old_h = ', h)
        #print('prev_x = ', prev_x)
        #print('xtp = ', xtp)
        
        old_h = h
        hg = self.NL( tf.matmul(old_h, Ug) + tf.matmul(x, Wg) + bg, 'sigmoid' )

        arg = tf.matmul(old_h, U2) + tf.matmul(x, W1) + tf.matmul(xtp, W2) + b1
        
        bx, bh = tf.matmul(x, B1), tf.matmul(h, B2) 
        
        for k in range(self.K):
            at1 = A1[k][self.t]
            h = h + at1 * hg * ( self.NL( tf.matmul(h, U1) + arg, self.sigma ) - h - old_h + bx + bh )
        
        #print('new_h = ', h)
        
        new_h = h #tf.concat([h1, h2], 1)
        paddings = tf.constant([[0, 0,], [0, REM]])
        hx = tf.pad(x, paddings)
        #print('hx = ', hx)
        #print('\n\n')
        
        #print('tf.concat([new_h, hx], 1) = ', tf.concat([new_h, hx], 1))
        new_state = tf.concat([new_h, hx], 1)
        
        self.t += 1
        if (self.t == self.n_timesteps): self.t == 0
        return new_h, new_state 
 

class B_TARNNCell(RNNCell):
    def __init__(self, hidden_size, n_features, n_timesteps, alpha_val=0.01, sigma='relu', K=2, gating=True):
        print("B_TARNNCell -> ", ' K=', K)
        super(B_TARNNCell, self).__init__()
        assert( n_features <= hidden_size )
        self.hidden_size = hidden_size
        self.sigma = sigma
        self.t = 0
        self.n_timesteps = n_timesteps
        self.n_features  = n_features
        self.K = K
        self.gating = gating
        self.alpha_val = alpha_val
        self.n_hidden = hidden_size 

    @property
    def state_size(self):
        return 2 * self.hidden_size

    @property
    def output_size(self):
        return self.hidden_size

    def NL(self, x, sigma='relu'):
        if sigma == 'relu': return tf.nn.relu(x)
        elif sigma == 'tanh': return tf.nn.tanh(x)
        elif sigma == 'sigmoid': return tf.nn.sigmoid(x)
        raise Exception('Non-linearity not found..')

    def call(self, x, h):
        #print('t = ', self.t, ' , n_timesteps = ', self.n_timesteps)
        alpha_val = self.alpha_val
        #initializer = tf.contrib.layers.xavier_initializer()
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32)
        bias_update_init = init_ops.constant_initializer(0.0, dtype=tf.float32)
        
        hidden_size = self.n_hidden #self.hidden_size
        n_features  = x.get_shape()[-1]  #self.n_features
        
        with vs.variable_scope("B_TARNNCell"):
            b1 = vs.get_variable("b1", [hidden_size], initializer=bias_update_init)
            b2 = vs.get_variable("b2", [hidden_size], initializer=bias_update_init)

            B1 = vs.get_variable("B1", [n_features, hidden_size], initializer=initializer)
            B2 = vs.get_variable("B2", [hidden_size, hidden_size], initializer=initializer)
            
            W1 = vs.get_variable("W1", [n_features, hidden_size], initializer=initializer)
            W2 = vs.get_variable("W2", [n_features, hidden_size], initializer=initializer)

            U1 = vs.get_variable("U1", [hidden_size, hidden_size], initializer=initializer)
            U2 = vs.get_variable("U2", [hidden_size, hidden_size], initializer=initializer)      

            Wg = vs.get_variable("Wg", [n_features, hidden_size], initializer=initializer)
            Ug = vs.get_variable("Ug", [hidden_size, hidden_size], initializer=initializer)        
            bg = vs.get_variable("bg", [hidden_size], initializer=init_ops.constant_initializer(-3.0, dtype=tf.float32))

            alpha_init = 0.01
            if alpha_val is not None: alpha_init = alpha_val
            beta_init = 1.0 - alpha_init
            #print('alpha_init = ', alpha_init)
            #print('beta_init = ', beta_init)
            #A1 = tf.Variable(alpha_init * tf.ones((self.K, self.n_timesteps)), dtype=tf.float32, name='alpha')
            
            A1 = vs.get_variable("alpha", [self.K, self.n_timesteps], 
                                 initializer=init_ops.constant_initializer(alpha_init, dtype=tf.float32))
        
        REM = self.hidden_size - self.n_features
        B = tf.shape(x)[0]
        
        #print('\n\n')
        h, prev_x = tf.split(value=h, num_or_size_splits=2, axis=1)
        xtp, _ = tf.split(prev_x, [self.n_features, REM], axis=1)
        
        #print('old_h = ', h)
        #print('prev_x = ', prev_x)
        #print('xtp = ', xtp)
        
        old_h = h
        arg = tf.matmul(old_h, U2) + tf.matmul(x, W1) + tf.matmul(xtp, W2) + b1
        
        bx, bh = tf.matmul(x, B1), tf.matmul(h, B2) 
        
        for k in range(self.K):
            at1 = A1[k][self.t]
            h = (1-at1) * h + at1 * ( self.NL( tf.matmul(h, U1) + arg, self.sigma ) - h - old_h + bx + bh )
            
        if self.gating:  
            hg = self.NL( tf.matmul(old_h, Ug) + tf.matmul(x, Wg) + bg, 'sigmoid' )
            h = (1-hg) * old_h + hg * h
        
        #print('new_h = ', h)
        
        new_h = h #tf.concat([h1, h2], 1)
        paddings = tf.constant([[0, 0,], [0, REM]])
        hx = tf.pad(x, paddings)
        #print('hx = ', hx)
        #print('\n\n')
        
        #print('tf.concat([new_h, hx], 1) = ', tf.concat([new_h, hx], 1))
        new_state = tf.concat([new_h, hx], 1)
        
        self.t += 1
        if (self.t == self.n_timesteps): self.t == 0
        return new_h, new_state 
 

'''
    dg1/dt = -(g1 + g2) + f( (U1 g1 + U2 g2) + (W1 x_t + W2 x_{t-1}) )
    dg2/dt = -(   + g2) + f( (      + U4 g2) + (W3 x_t + W4 x_{t-1}) )
'''
class iRNN_DualODECell(RNNCell):
    def __init__(self, hidden_size, n_features, n_timesteps, alpha_val=0.01, sigma='relu', K=2, gating=True):
        super(iRNN_DualODECell, self).__init__()
        assert( n_features <= hidden_size )
        self.hidden_size = hidden_size
        self.sigma = sigma
        self.t = 0
        self.n_timesteps = n_timesteps
        self.n_features  = n_features
        self.K1 = K
        self.K2 = 1
        self.gating = gating
        self.alpha_val = alpha_val
        self.n_hidden = hidden_size // 2
        assert( hidden_size % 2 == 0 )
        print("iRNN_DualODE -> ", ' K1=', self.K1, ', K2=', self.K2)

    @property
    def state_size(self):
        return 2 * self.hidden_size

    @property
    def output_size(self):
        return self.hidden_size

    def NL(self, x, sigma='relu'):
        if sigma == 'relu': return tf.nn.relu(x)
        elif sigma == 'tanh': return tf.nn.tanh(x)
        elif sigma == 'sigmoid': return tf.nn.sigmoid(x)
        raise Exception('Non-linearity not found..')

    def call(self, x, h):
        #print('t = ', self.t, ' , n_timesteps = ', self.n_timesteps)
        alpha_val = self.alpha_val
        #initializer = tf.contrib.layers.xavier_initializer()
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32)
        bias_update_init = init_ops.constant_initializer(0.0, dtype=tf.float32)
        
        hidden_size = self.n_hidden #self.hidden_size
        n_features  = x.get_shape()[-1]  #self.n_features
        
        with vs.variable_scope("iRNN_DualODECell"):
            b1 = vs.get_variable("b1", [hidden_size], initializer=bias_update_init)
            b2 = vs.get_variable("b2", [hidden_size], initializer=bias_update_init)

            W1 = vs.get_variable("W1", [n_features, hidden_size], initializer=initializer)
            W2 = vs.get_variable("W2", [n_features, hidden_size], initializer=initializer)
            W3 = vs.get_variable("W3", [n_features, hidden_size], initializer=initializer)
            W4 = vs.get_variable("W4", [n_features, hidden_size], initializer=initializer)

            U1 = vs.get_variable("U1", [hidden_size, hidden_size], initializer=initializer)
            U2 = vs.get_variable("U2", [hidden_size, hidden_size], initializer=initializer)
            U4 = vs.get_variable("U4", [hidden_size, hidden_size], initializer=initializer)        

            Wg = vs.get_variable("Wg", [n_features, hidden_size], initializer=initializer)
            Ug = vs.get_variable("Ug", [hidden_size, hidden_size], initializer=initializer)        
            bg = vs.get_variable("bg", [hidden_size], initializer=init_ops.constant_initializer(-3.0, dtype=tf.float32))
            #bg = vs.get_variable("bg", [hidden_size], initializer=init_ops.constant_initializer(+1.0, dtype=tf.float32))

            alpha_init = 0.01
            if alpha_val is not None: alpha_init = alpha_val
            beta_init = 1.0# - alpha_init
            
            A1 = vs.get_variable("alpha",  [self.K1, self.n_timesteps], 
                                 initializer=init_ops.constant_initializer(alpha_init, dtype=tf.float32))
            A2 = vs.get_variable("alpha2", [self.K2, self.n_timesteps], 
                                 initializer=init_ops.constant_initializer(alpha_init, dtype=tf.float32))
            B1 = vs.get_variable("beta",  [self.K1, self.n_timesteps], 
                                 initializer=init_ops.constant_initializer(beta_init, dtype=tf.float32))
            B2 = vs.get_variable("beta2", [self.K2, self.n_timesteps], 
                                 initializer=init_ops.constant_initializer(beta_init, dtype=tf.float32))
            
        
        REM = self.hidden_size - self.n_features
        B = tf.shape(x)[0]
        
        g12, prev_x = tf.split(value=h, num_or_size_splits=2, axis=1)
        g1, g2 = tf.split(value=g12, num_or_size_splits=2, axis=1)
        xtp, _ = tf.split(prev_x, [self.n_features, REM], axis=1)
        
        old_g1, old_g2 = g1, g2
        
        new_wx1, new_wx3 = tf.matmul(x, W1), tf.matmul(x, W3)
        new_wx2, new_wx4 = tf.matmul(xtp, W2), tf.matmul(xtp, W4)
        
        #alpha2 = tf.matmul(old_g2, U4) + new_wx3 + new_wx4 + b2
        #alpha1 = tf.matmul(old_g1, U1) + new_wx1 + new_wx2 + b1
        
        alpha2 = tf.matmul(old_g2, U4) + new_wx3 + b2
        alpha1 = tf.matmul(old_g1, U1) + new_wx1 + b1
        
        at2 = A2[0][self.t]
        bt2 = B2[0][self.t]
        g2 = bt2 * g2 + at2 * ( self.NL( alpha2, self.sigma ) )
        
        '''
        for k in range(self.K2):
            at2 = A2[k][self.t]
            bt2 = B2[k][self.t]
            new_ug42 = tf.matmul(g2, U4)
            g2 = bt2 * g2 + at2 * ( self.NL( new_ug42 + alpha2, self.sigma ) - old_g2 )
        '''
        
        new_ug22 = tf.matmul(g2, U2)
        for k in range(self.K1):
            new_ug11 = tf.matmul(g1, U1)
            at1 = A1[k][self.t]
            bt1 = 1-at1 #B1[k][self.t]
            g1 = bt1 * g1 + at1 * ( self.NL( new_ug11 + new_ug22 + alpha1, self.sigma ) - g2 - old_g1 )
            
        if self.gating:  
            hg = self.NL( tf.matmul(old_g2, Ug) + tf.matmul(x, Wg) + bg, 'sigmoid' )
            #hg = self.NL( tf.matmul(g2, Ug) + tf.matmul(x, Wg) + bg, 'sigmoid' )
            #g2 = (1-hg) * old_g2 + hg * g2
            g1 = (1-hg) * old_g1 + hg * g1
            #g1 = (1-hg) * g2 + hg * g1
            
        h1, h2 = g1, g2
        
        new_h = tf.concat([h1, h2], 1)
        paddings = tf.constant([[0, 0,], [0, REM]])
        hx = tf.pad(x, paddings)
        new_state = tf.concat([h1, h2, hx], 1)
        
        self.t += 1
        if (self.t == self.n_timesteps): self.t == 0
        return new_h, new_state  
