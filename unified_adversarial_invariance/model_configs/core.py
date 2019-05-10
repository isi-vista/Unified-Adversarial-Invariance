class ModelConfigBase(object):
    
    def __init__(self):
        self.nclasses = None
        self.nz = None
        self.x_shape = None
        self.embedding_dim_1 = None
        self.embedding_dim_2 = None
        self.embedding_activation = None
        self.predictor_loss = None
        self.decoder_loss = None
        self.disentangler_loss = None
        self.z_discriminator_loss = None
    
    def encoder(self, name='encoder'):
        raise NotImplementedError
    
    def noisy_transformer(self, params=[], name='noisy_transformer'):
        raise NotImplementedError
    
    def predictor(self, name='predictor'):
        raise NotImplementedError
    
    def decoder(self, name='decoder'):
        raise NotImplementedError
    
    def disentangler(self, input_dim=None, output_dim=None, name='disentangler'):
        raise NotImplementedError
    
    def z_discriminator(self, name='z_discriminator'):
        raise NotImplementedError
