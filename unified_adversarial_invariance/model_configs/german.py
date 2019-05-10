from unified_adversarial_invariance.model_configs.core import ModelConfigBase
from unified_adversarial_invariance.utils.training_utils \
    import disentanglement_loss

from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model


class ModelConfig(ModelConfigBase):
    
    def __init__(self):
        super(ModelConfig, self).__init__()
        self.nclasses = 2
        self.nz = 2
        self.x_shape = (60,)
        self.embedding_dim_1 = 64
        self.embedding_dim_2 = 64
        self.embedding_activation = 'tanh'
        self.predictor_loss = 'binary_crossentropy'
        self.decoder_loss = 'mean_squared_error'
        self.disentangler_loss = disentanglement_loss
        self.z_discriminator_loss = 'binary_crossentropy'
    
    def encoder(self, name='encoder'):
        x = Input(self.x_shape, name='encoder_input')
        e1 = Dense(
                self.embedding_dim_1,
                activation=self.embedding_activation,
                name='encoder_fc1'
            )(x)
        e2 = Dense(
                self.embedding_dim_2,
                activation=self.embedding_activation,
                name='encoder_fc2'
            )(x)
        
        return Model(inputs=[x], outputs=[e1, e2], name=name)
    
    def noisy_transformer(self, params=[0.5], name='noisy_transformer'):
        dropout_rate = params[0]
        return Dropout(dropout_rate)
    
    def predictor(self, name='predictor'):
        e1 = Input((self.embedding_dim_1,))
        h = BatchNormalization(name='predictor_bn1')(e1)
        y = Dense(1, activation='sigmoid', name='predictor_output')(h)
        
        return Model(e1, y, name=name)
    
    def decoder(self, name='decoder'):
        e1 = Input((self.embedding_dim_1,))
        e2 = Input((self.embedding_dim_2,))
        e = Concatenate()([e1, e2])
        x = Dense(self.x_shape[0], activation='relu')(e)
        
        return Model(inputs=[e1, e2], outputs=[x], name=name)
    
    def disentangler(self, input_dim=None, output_dim=None, name='disentangler'):
        if input_dim is None:
            input_dim = self.embedding_dim_2
        if output_dim is None:
            output_dim = self.embedding_dim_1
        
        ei = Input((input_dim,), name='disentangler_input')
        ej = Dense(
                output_dim, activation=self.embedding_activation,
                name='disentangler_output'
            )(ei)
        
        return Model(ei, ej, name=name)
        
    def z_discriminator(self, name='z_discriminator'):
        e1 = Input((self.embedding_dim_1,), name='z_discriminator_input')
        h = Dense(
                64, activation='relu',
                name='z_discriminator_fc1'
            )(e1)
        h = Dense(
                64, activation='relu',
                name='z_discriminator_fc2'
            )(h)
        y = Dense(
                1, activation='sigmoid',
                name='z_discriminator_output'
            )(h)
        
        return Model(e1, y, name=name)
