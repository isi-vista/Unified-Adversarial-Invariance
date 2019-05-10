from keras.losses import mean_squared_error
from keras.callbacks import Callback

from joblib import Parallel
from joblib import delayed
import shutil
import os


def disentanglement_loss(y_true, y_pred):
    embedding_dim = y_pred.shape[-1].value / 2
    e_i_true = y_pred[:, :embedding_dim]
    e_i_pred = y_pred[:, embedding_dim:]
    
    return mean_squared_error(e_i_true, e_i_pred)


class ModelsCheckpoint(Callback):

    def __init__(self, models, filepaths, verbose=0,
                 save_weights_only=False, period=1, remote_path=None):
        super(ModelsCheckpoint, self).__init__()
        self.verbose = verbose
        self.models = models
        self.filepaths = filepaths
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        remote_path = remote_path or os.path.dirname(self.filepaths[0])

    def __save_models(self, epoch):
        for model, filepath in zip(self.models, self.filepaths):
            filepath = filepath.format(epoch)
            if self.save_weights_only:
                model.save_weights(filepath, overwrite=True)
            else:
                model.save(filepath, overwrite=True)
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            if self.verbose > 0:
                print('Epoch %05d: saving models' % (epoch))
            self.__save_models(epoch)


class Sync(Callback):
    '''Moves files from source to destination at specified frequency.'''

    def __init__(self, src, dest, period=1, num_jobs=1):
        super(Sync, self).__init__()
        self.src = src
        self.dest = dest
        self.period = period
        self.epochs_since_last_bkp = 0
        self.num_jobs = num_jobs

    def sync(self):
        copy_files = [os.path.join(self.src, f) for f in os.listdir(self.src)]
        Parallel(backend='threading', n_jobs=self.num_jobs)(
            delayed(shutil.copy)(f, self.dest) for f in copy_files
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_bkp += 1
        if self.epochs_since_last_bkp >= self.period:
            self.epochs_since_last_bkp = 0
            self.sync()


def prepare_callbacks(models, model_names, optimizers,
                      weights_path, remote_weights_path, sync_frequency,
                      encoder, dataset, data_loading_kwargs,
                      streaming_data=False):
    checkpoint_cb = ModelsCheckpoint(
        models,
        [
            os.path.join(weights_path, '%s-{:05d}.h5' % s) \
                for s in model_names
        ],
        save_weights_only=True, period=5, verbose=1,
        remote_path=remote_weights_path
    )
    sync_cb = Sync(
        weights_path, remote_weights_path,
        period=sync_frequency, num_jobs=5
    )

    callbacks = [checkpoint_cb, sync_cb]

    return callbacks, sync_cb
