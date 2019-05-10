import numpy


def create_fake_labels(labels):
    return numpy.random.permutation(labels)

def get_targets_adversarial(x, y, z=None, bias=False,
                            embedding_dim_1=None, embedding_dim_2=None):
    predictor_target = y
    decoder_target = x
    
    # dummy disentangler targets to satisfy Keras loss API
    disentangler_targets = [
        numpy.zeros((x.shape[0], embedding_dim_1 * 2), dtype='float32'),
        numpy.zeros((x.shape[0], embedding_dim_2 * 2), dtype='float32')
    ]
    
    targets_aux = [predictor_target, decoder_target] + disentangler_targets
    
    # prepare targets to satisfy keras_adversarial API
    targets = []
    if bias:
        z_true = z
        z_fake = create_fake_labels(z_true)
        targets = targets_aux + [z_fake] + targets_aux + [z_true]
    else:
        targets = targets_aux + targets_aux
    
    return targets


def get_targets(x, y, z=None, bias=False, y_only=False,
                z_only=False, embedding_dim_1=None, embedding_dim_2=None):
    if y_only:
        return y
    elif z_only:
        return z
    else:
        targets = get_targets_adversarial(
            x, y, z=z, bias=bias,
            embedding_dim_1=embedding_dim_1,
            embedding_dim_2=embedding_dim_2
        )
    
    return targets


class DatasetBase(object):
    
    def __init__(self, fold_id=None):
        self.fold_id = fold_id
        
        self.data_file = None
        self.data = None
        self.generator_steps = {
            'train': None,
            'valid': None,
            'test': None
        }
    
    def __load_data(*args, **kwargs):
        raise NotImplementedError
    
    def get_data(self, split, bias=True, y_only=False, z_only=False,
                 embedding_dim_1=None, embedding_dim_2=None):
        raise NotImplementedError
    
    def get_generator(self, split, bias=True, y_only=False, z_only=False,
                      embedding_dim_1=None, embedding_dim_2=None):
        raise NotImplementedError
