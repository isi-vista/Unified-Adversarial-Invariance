from unified_adversarial_invariance.datasets.core import DatasetBase
from unified_adversarial_invariance.datasets.core import get_targets

from keras.preprocessing.image import transform_matrix_offset_center
from keras.preprocessing.image import apply_transform
from keras.utils import to_categorical
from keras.datasets import mnist

import numpy


class MNIST_DataGenerator(object):
    
    def __init__(self, x, y, mode='train', batch_size=64,
                 theta=[-45.0, -22.5, 0.0, 22.5, 45.0],
                 nb_batches_per_epoch=1000, seed=6789,
                 bias=False, y_only=False, z_only=False,
                 embedding_dim_1=None, embedding_dim_2=None):
        self.image_data = x
        self.labels = y
        self.nb_samples = y.shape[0]
        self.batch_size = batch_size
        self.theta = theta
        self.nz = len(self.theta)
        self.nb_batches_per_epoch = nb_batches_per_epoch
        self.mode = mode
        self.seed = seed
        self.epoch = 0
        self.idx = 0
        self._reset_prng(seed)
        self.angle_class_map = {a: i for i, a in enumerate(self.theta)}
        self.bias = bias
        self.y_only = y_only
        self.z_only = z_only
        self.embedding_dim_1 = embedding_dim_1
        self.embedding_dim_2 = embedding_dim_2
    
    def _reset_prng(self, seed=None):
        if seed is None:
            self.prng = numpy.random.RandomState(self.seed)
        else:
            self.prng = numpy.random.RandomState(seed)
        return

    def _rotate_image(self, x3d, theta):
        theta = numpy.deg2rad(theta)
        rotation_matrix = numpy.array(
            [
                [numpy.cos(theta), -numpy.sin(theta), 0],
                [numpy.sin(theta), numpy.cos(theta), 0],
                [0, 0, 1]
            ]
        )
        h, w, d = x3d.shape
        transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
        xrot = apply_transform(x3d, transform_matrix, 2, 'nearest', 0)
        return xrot

    def _get_one_batch(self, batch_idx):
        if self.mode == 'train':
            indices = self.prng.randint(0, self.nb_samples, self.batch_size)
        else:
            indices = numpy.arange(
                        batch_idx * self.batch_size,
                        (batch_idx + 1) * self.batch_size
                      ) % self.nb_samples
            if self.seed is not None:
                seed = self.seed + (batch_idx % self.nb_batches_per_epoch)
            else:
                seed = batch_idx % self.nb_batches_per_epoch
            self._reset_prng(seed)

        indices = indices.tolist()
        X = self.image_data[indices]
        Y = self.labels[indices]

        # data augmentation
        angles = numpy.random.choice(self.theta, self.batch_size)
        
        XR = []
        for ii, (x, angle) in enumerate(zip(X, angles)):
            xr = self._rotate_image(x, angle)
            XR.append(numpy.expand_dims(xr, axis=0))
        XR = numpy.concatenate(XR, axis=0)
        
        Z = numpy.array([self.angle_class_map[angle] for angle in angles])
        Z = to_categorical(Z, self.nz)

        return (XR, Y, Z)

    def __getitem__(self, batch_idx):
        x, y, z = self._get_one_batch(batch_idx)
        
        targets = get_targets(
            x, y, z=z,
            bias=self.bias, y_only=self.y_only, z_only=self.z_only,
            embedding_dim_1=self.embedding_dim_1,
            embedding_dim_2=self.embedding_dim_2
        )
        
        return (x, targets)

    def __iter__(self):
        return self

    def next(self):
        idx = self.idx
        if idx >= self.nb_batches_per_epoch:
            idx = 0
            self.epoch += 1
        self.idx = idx + 1
        return self[idx]


class Dataset(DatasetBase):
    
    def __init__(self, fold_id=None):
        super(Dataset, self).__init__()
        self.generator_steps = {
            'train': 500,
            'valid': 75,
            'test': 156
        }
        self.__load_data()
    
    def __load_data(self, seed=123456):
        (x, y), (x_test, y_test) = mnist.load_data()
        num_classes = 10

        x = x.reshape(60000, 28, 28, 1)
        x_test = x_test.reshape(10000, 28, 28, 1)
        x = x.astype('float32')
        x_test = x_test.astype('float32')
        x = x / 255
        x_test = x_test / 255

        # convert class vectors to binary class matrices
        y = to_categorical(y, num_classes)
        y_test = to_categorical(y_test, num_classes)

        # seperate a validation set from training data
        indices = range(60000)
        numpy.random.seed(seed)
        numpy.random.shuffle(indices)
        indices_train = indices[:50000]
        indices_valid = indices[50000:]
        x_train, x_valid = x[indices_train], x[indices_valid]
        y_train, y_valid = y[indices_train], y[indices_valid]
    
        self.data = {
            'train': (x_train, y_train),
            'valid': (x_valid, y_valid),
            'test': (x_test, y_test)
        }
    
    def get_generator(self, split, bias=True, y_only=False, z_only=False,
                      embedding_dim_1=None, embedding_dim_2=None):
        
        x, y = self.data[split]
        
        dgen = MNIST_DataGenerator(
            x, y, mode=split,
            theta=[-45.0, -22.5, 0.0, 22.5, 45.0],
            bias=bias, y_only=y_only, z_only=z_only,
            embedding_dim_1=embedding_dim_1,
            embedding_dim_2=embedding_dim_2
        )
        
        return dgen
