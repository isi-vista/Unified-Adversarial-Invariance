from unified_adversarial_invariance.model_configs.core import ModelConfigBase
from unified_adversarial_invariance.utils.training_utils \
    import disentanglement_loss

from keras.layers import BatchNormalization
from keras.layers import UpSampling2D
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model


class ModelConfig(ModelConfigBase):
    
    def __init__(self):
        super(ModelConfig, self).__init__()
        self.nclasses = 10
        self.nz = 2
        self.x_shape = (28, 28, 1)
        self.embedding_dim_1 = 10
        self.embedding_dim_2 = 20
        self.nz = 'tanh'
        self.predictor_loss = 'categorical_crossentropy'
        self.decoder_loss = 'mean_squared_error'
        self.disentangler_loss = disentanglement_loss
    
    def encoder(self, name='encoder'):
        x = Input(self.x_shape, name='encoder_input')
        
        h = Conv2D(
                64, (5, 5), strides=(2, 2), padding='same', name='encoder_conv1'
            )(x)
        h = BatchNormalization(name='encoder_bn1')(h)
        h = Activation('relu', name='encoder_relu1')(h)
        h = Flatten(name='flatten')(h)
        
        e1 = Dense(
                self.embedding_dim_1, name='embedding_1', activation=self.nz
             )(h)
        e2 = Dense(
                self.embedding_dim_2, name='embedding_2', activation=self.nz
             )(h)
        
        return Model(inputs=[x], outputs=[e1, e2], name=name)
    
    def noisy_transformer(self, params=[0.5], name='noisy_transformer'):
        dropout_rate = params[0]
        return Dropout(dropout_rate)
    
    def predictor(self, name='predictor'):
        e1 = Input((self.embedding_dim_1,), name='predictor_input')
        h = BatchNormalization(name='predictor_bn1')(e1)

        h = Dense(128, name='predictor_fc2')(h)
        h = BatchNormalization(name='predictor_bn2')(h)
        h = Activation('relu', name='predictor_relu2')(h)

        y = Dense(self.nclasses, activation='softmax', name='predictor_output')(h)

        return Model(e1, y, name=name)
    
    def decoder(self, name='decoder'):
        x_height, x_width, x_channels = self.x_shape
        x_height_half = x_height / 2
        x_width_half = x_width / 2
        
        e1 = Input((self.embedding_dim_1,))
        e2 = Input((self.embedding_dim_2,))
        e = Concatenate()([e1, e2])
        
        h = Dense((256 * x_height_half * x_width_half), name='decoder_conv1')(e)
        h = BatchNormalization(name='decoder_bn1')(h)
        h = Activation('relu', name='decoder_relu1')(h)
        h = Reshape((x_height_half, x_width_half, 256))(h)
        
        h = UpSampling2D(size=(2, 2))(h)
        h = Conv2D(128, (3, 3), name='decoder_conv2', padding='same')(h)
        h = BatchNormalization()(h)
        h = Activation('relu', name='decoder_relu2')(h)
        
        h = Conv2D(x_channels, (1, 1), name='decoder_conv4', padding='same')(h)
        x = Activation('sigmoid')(h)
        x = Reshape(self.x_shape, name='decoder_output')(x)
        
        return Model(inputs=[e1, e2], outputs=[x], name=name)
    
    def disentangler(self, input_dim=None, output_dim=None, name='disentangler'):
        if input_dim is None:
            input_dim = self.embedding_dim_2
        if output_dim is None:
            output_dim = self.embedding_dim_1

        ei = Input((input_dim,), name='disentangler_input')
        ej = Dense(
                output_dim, activation=self.nz,
                name='disentangler_output'
            )(ei)
        
        return Model(ei, ej, name=name)
