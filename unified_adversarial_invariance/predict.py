from unified_adversarial_invariance.model_configs import MODEL_CONFIGS_DICT
from unified_adversarial_invariance.datasets import DATASETS_DICT
from unified_adversarial_invariance.unifai import UnifAI_Config
from unified_adversarial_invariance.unifai import UnifAI

import argparse
import numpy
import os


def get_predictions(model_config, remote_weights_path, checkpoint_epoch,
                    dataset, split, streaming_data=True):
    
    # Create config
    
    config = UnifAI_Config()
    config.model_config = model_config
    config.remote_weights_path = remote_weights_path
    
    # Build model
    
    unifai = UnifAI(config)
    unifai.build_model_inference(checkpoint_epoch=checkpoint_epoch)
    
    # Set up data configuration
    
    data_loading_kwargs = {
        'embedding_dim_1': model_config.embedding_dim_1,
        'embedding_dim_2': model_config.embedding_dim_2
    }
    
    # Get data
    
    data = None
    prediction_steps = None
    if streaming_data:
        data = dataset.get_generator(split, **data_loading_kwargs)
        prediction_steps = dataset.generator_steps[split]
    else:
        data = dataset.get_data(split, **data_loading_kwargs)[0]
    
    # Get predictions
    
    predictions = unifai.predict(
        data, streaming_data=streaming_data, prediction_steps=prediction_steps
    )
    
    return predictions


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
        'checkpoint_epoch', type=int,
        help='checkpoint epoch'
    )
    parser.add_argument(
        'split', type=str,
        help='train, valid, or test'
    )
    parser.add_argument(
        'output_path', type=str,
        help='path for saving predictions'
    )
    parser.add_argument(
        '-f', '--fold-id', type=int, default=-1,
        help='fold_id in case of cross-validation'
    )
    parser.add_argument(
        '-s', '--streaming-data', type=int, default=0,
        help='train using static (0) or streaming (1) data'
    )
    args = parser.parse_args()

    # Resolve command-line arguments
    
    streaming_data = args.streaming_data == 1

    # Load data util and config for network-modules
    
    model_config = MODEL_CONFIGS_DICT[args.data_name].ModelConfig()
    fold_id = args.fold_id
    if fold_id == -1:
        fold_id = None
    dataset = DATASETS_DICT[args.data_name].Dataset(fold_id=fold_id)
    
    # Create local and remote paths
    
    remote_weights_path = os.path.join(args.weights_root, args.model_name)

    # Get predictions
    
    predictions = get_predictions(
        model_config, remote_weights_path, args.checkpoint_epoch,
        dataset, args.split, streaming_data=streaming_data
    )
    
    # Save predictions
    
    numpy.save(args.output_path, predictions)
    print '\nPredictions saved at %s \n' % args.output_path


if __name__ == '__main__':
    main()
