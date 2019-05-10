from unified_adversarial_invariance.utils.training_utils import prepare_callbacks
from unified_adversarial_invariance.model_configs import MODEL_CONFIGS_DICT
from unified_adversarial_invariance.datasets import DATASETS_DICT
from unified_adversarial_invariance.unifai import UnifAI_Config
from unified_adversarial_invariance.unifai import UnifAI

from keras.optimizers import Adam

import argparse
import os


def train_model(model_config, dataset, adversarial_training_schedule,
                local_weights_path, remote_weights_path, sync_frequency,
                dropout_rate=0.5, bias=False, predictor_loss_weight=100.,
                decoder_loss_weight=0.1, disentangler_loss_weight=1.,
                z_discriminator_loss_weight=10., streaming_data=True,
                batch_size=64, epochs=10000, gpus=1):
    
    # Create config
    
    config = UnifAI_Config()
    
    config.model_config = model_config
    config.dropout_rate = dropout_rate
    config.bias = bias
    
    config.local_weights_path = local_weights_path
    config.remote_weights_path = remote_weights_path
    config.sync_frequency = sync_frequency
    
    config.losses = [
        model_config.predictor_loss, model_config.decoder_loss,
        model_config.disentangler_loss, model_config.disentangler_loss
    ]
    config.loss_weights = [
        predictor_loss_weight, decoder_loss_weight,
        disentangler_loss_weight, disentangler_loss_weight
    ]
    if bias:
        config.losses.append(model_config.z_discriminator_loss)
        config.loss_weights.append(z_discriminator_loss_weight)
    
    config.main_lr = 1e-4
    config.adv_lr = 1e-3
    config.optimizers = [
        Adam(config.main_lr, decay=1e-4), Adam(config.adv_lr, decay=1e-4)
    ]
    
    config.metrics = {'y': 'accuracy'}
    if bias:
        config.metrics['z'] = 'accuracy'
    
    config.training_schedule = adversarial_training_schedule
    
    config.num_gpus = gpus
    config.batch_size = batch_size
    
    # Build model
    
    unifai = UnifAI(config)
    unifai.build_model_train()
    unifai.compile_model()
    
    # Set up data configuration
    
    data_loading_kwargs = {
        'embedding_dim_1': model_config.embedding_dim_1,
        'embedding_dim_2': model_config.embedding_dim_2,
        'bias': bias
    }
    
    # Get data
    
    dtrain = None
    dvalid = None
    training_steps = None
    validation_steps = None
    if streaming_data:
        dtrain = dataset.get_generator('train', **data_loading_kwargs)
        dvalid = dataset.get_generator('valid', **data_loading_kwargs)
        training_steps = dataset.generator_steps['train']
        validation_steps = dataset.generator_steps['valid']
    else:
        dtrain = dataset.get_data('train', **data_loading_kwargs)
        dvalid = dataset.get_data('valid', **data_loading_kwargs)
    
    # Prepare training callbacks
    
    chkpt_models = [
        unifai.encoder, unifai.predictor, unifai.decoder,
        unifai.disentangler1, unifai.disentangler2
    ]
    chkpt_model_names = [
        'encoder', 'predictor', 'decoder',
        'disentangler1', 'disentangler2'
    ]
    if bias:
        chkpt_models.append(unifai.z_discriminator)
        chkpt_model_names.append('z_discriminator')
    
    callbacks, sync_callback = prepare_callbacks(
        chkpt_models, chkpt_model_names, unifai.optimizers,
        local_weights_path, remote_weights_path, sync_frequency,
        unifai.encoder, dataset, data_loading_kwargs,
        streaming_data=streaming_data
    )
    
    # Train
    unifai.fit(
        dtrain, dvalid,
        streaming_data=streaming_data, epochs=epochs, callbacks=callbacks,
        training_steps=training_steps, validation_steps=validation_steps
    )
    
    # Sync local files with remote directory at the end of training
    sync_callback.sync()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_name', type=str,
        help='name of dataset'
    )
    parser.add_argument(
        'model_name', type=str,
        help='model name'
    )
    parser.add_argument(
        'weights_root', type=str,
        help='root dir of (remote) path to dump model weights'
    )
    parser.add_argument(
        '-dr', '--dropout-rate', type=float, default=0.5,
        help='dropout-rate for e1 --> decoder'
    )
    parser.add_argument(
        '-b', '--bias', type=int, default=0,
        help='add supervised z-discriminator (1) or not (0)'
    )
    parser.add_argument(
        '-f', '--fold-id', type=int, default=-1,
        help='fold_id in case of cross-validation'
    )
    parser.add_argument(
        '-s', '--streaming-data', type=int, default=0,
        help='train using static (0) or streaming (1) data'
    )
    parser.add_argument(
        '-ats', '--adversarial-training-schedule', type=str, default='01111111111',
        help='adversarial training schedule'
    )
    parser.add_argument(
        '-plw', '--predictor-loss-weight', type=float, default=100.,
        help='predictor loss weight or alpha'
    )
    parser.add_argument(
        '-declw', '--decoder-loss-weight', type=float, default=0.1,
        help='decoder loss weight or beta'
    )
    parser.add_argument(
        '-dislw', '--disentangler-loss-weight', type=float, default=1.,
        help='disentangler loss weight or gamma'
    )
    parser.add_argument(
        '-zlw', '--z-discriminator-loss-weight', type=float, default=1.,
        help='z_discriminator loss weight or delta'
    )
    parser.add_argument(
        '-e', '--epochs', type=int, default=10000,
        help='number of epochs to train'
    )
    parser.add_argument(
        '-g', '--gpus', type=int, default=1,
        help='number of GPUs for data parallelism'
    )
    parser.add_argument(
        '-bs', '--batch-size', type=int, default=64,
        help='batch-size (per GPU)'
    )
    parser.add_argument(
        '-sf', '--sync-frequency', type=int, default=50,
        help='number of epochs to wait between local and remote directory syncs'
    )
    args = parser.parse_args()

    # Resolve command-line arguments
    
    streaming_data = args.streaming_data == 1
    bias = args.bias == 1

    # Load data util and config for network-modules
    
    model_config = MODEL_CONFIGS_DICT[args.data_name].ModelConfig()
    fold_id = args.fold_id
    if fold_id == -1:
        fold_id = None
    dataset = DATASETS_DICT[args.data_name].Dataset(fold_id=fold_id)
    
    # Create local and remote paths
    
    local_weights_path = os.path.join(os.getenv('TMPDIR'), args.model_name)
    remote_weights_path = os.path.join(args.weights_root, args.model_name)
    for path in [local_weights_path, remote_weights_path]:
        assert os.path.exists(os.path.dirname(path)), \
            'Cannot create directory %s' % path
        if not os.path.exists(path):
            os.makedirs(path)
    
    # Train model

    train_model(
        model_config, dataset, args.adversarial_training_schedule,
        local_weights_path, remote_weights_path, args.sync_frequency,
        dropout_rate=args.dropout_rate,
        bias=bias,
        predictor_loss_weight=args.predictor_loss_weight,
        decoder_loss_weight=args.decoder_loss_weight,
        disentangler_loss_weight=args.disentangler_loss_weight,
        z_discriminator_loss_weight=args.z_discriminator_loss_weight,
        streaming_data=streaming_data,
        batch_size=args.batch_size,
        epochs=args.epochs,
        gpus=args.gpus
    )


if __name__ == '__main__':
    main()
