from keras_adversarial.adversarial_utils import uniform_latent_sampling
from keras_adversarial import AdversarialOptimizerScheduled
from keras_adversarial import AdversarialModel
from keras_adversarial import fix_names

from keras.utils import multi_gpu_model
from keras.layers import Concatenate
from keras.models import Model

import tensorflow as tf

import os


class UnifAI_Config(object):
    
    def __init__(self):
        self.model_config = None
        self.bias = False
        self.dropout_rate = -1.0
        
        self.local_weights_path = ''
        self.remote_weights_path = ''
        self.sync_frequency = -1
        
        self.losses = []
        self.loss_weights = []
        self.main_lr = -1.0
        self.adv_lr = -1.0
        self.optimizers = []
        self.metrics = {}
        
        self.training_schedule = ''
        
        self.num_gpus = -1
        self.batch_size = -1


class UnifAI_ModuleBuilder(object):
    
    def __init__(self, model_config, weights_dir):
        self.model_config = model_config
        self.weights_dir = weights_dir
        self.build_functions = {
            'encoder': self.model_config.encoder,
            'predictor': self.model_config.predictor,
            'decoder': self.model_config.decoder,
            'noisy_transformer': self.model_config.noisy_transformer,
            'disentangler': self.model_config.disentangler,
        }
        try:
            self.build_functions['z_discriminator'] = \
                self.model_config.z_discriminator
        except AttributeError:
            pass

    def build_module(self, module_type, name=None,
                     build_kwargs={}, epoch=None, load_weights=True):
        # Default model-name is the same as model-type
        if name is None:
            name = module_type

        # Build module
        module = self.build_functions[module_type](name=name, **build_kwargs)
        

        # Try to load weights
        if not isinstance(module, Model):
            load_weights = False
        
        if load_weights:
            try:
                weights_path = None
                if epoch is None or epoch == 'latest':
                    weights_path = os.path.join(self.weights_dir, '%s.h5' % name)
                    epoch = 'latest'
                else:
                    weights_path = os.path.join(
                        self.weights_dir, ('%s-{:05d}.h5' % name).format(epoch)
                    )
                module.load_weights(weights_path)
                print '*** %s loaded from %s ***' % (name, epoch)
            except:
                if epoch is not None and epoch != 'latest':
                    raise ValueError, 'Weights not found for epoch: %s' % epoch
                else:
                    pass

        return module

    def build_default_modules(self, module_types, load_weights=True):
        modules = [
            self.build_module(
                mt, load_weights=load_weights
            ) for mt in module_types
        ]
        return modules


class UnifAI(object):
    
    def __init__(self, config):
        self.config = config
        self.model_config = self.config.model_config  # alias
        
        self.module_builder = UnifAI_ModuleBuilder(
            self.model_config, self.config.remote_weights_path
        )
        
        self.encoder = None
        self.predictor = None
        self.decoder = None
        self.noisy_transformer = None
        self.disentangler1 = None
        self.disentangler2 = None
        self.z_discriminator = None
        
        self.model_inference = None
        self.model_train = None
        
        self.optimizers = self.config.optimizers  # alias
        self.compiled = False
    
    def __random_target(self, x, dim, embedding_activation):
            range_low = -1 if embedding_activation == 'tanh' else 0
            range_high = 1
            tfake = uniform_latent_sampling(
                (dim,), low=range_low, high=range_high
            )(x)
            return tfake
    
    def __build_connected_network_train(self, main=True):
        x = self.encoder.inputs[0]
        e1, e2 = self.encoder(x)
        
        noisy_e1 = self.noisy_transformer(e1)

        y = self.predictor(e1)
        x_pred = self.decoder([noisy_e1, e2])

        e1_dim = int(self.encoder.outputs[0].shape[-1])
        e2_dim = int(self.encoder.outputs[1].shape[-1])

        output_vars = [y, x_pred]
        output_names = ['y', 'x_pred']

        e1_target = e1
        e2_target = e2
        e2_pred = self.disentangler1(e1)
        e1_pred = self.disentangler2(e2)

        if main:
            embedding_activation = self.model_config.embedding_activation
            e2_target = self.__random_target(x, e2_dim, embedding_activation)
            e1_target = self.__random_target(x, e1_dim, embedding_activation)

        e1_e1_pred = Concatenate()([e1_target, e1_pred])
        output_vars.append(e1_e1_pred)
        output_names.append('e1pred')
        
        e2_e2_pred = Concatenate()([e2_target, e2_pred])
        output_vars.append(e2_e2_pred)
        output_names.append('e2pred')
        
        if self.z_discriminator is not None:
            z = self.z_discriminator(e1)
            output_vars.append(z)
            output_names.append('z')

        outputs = fix_names(output_vars, output_names)
        network = Model(inputs=[x], outputs=outputs)

        return network
    
    def build_model_train(self):
        if self.model_train is None:
            with tf.device('/gpu:0'):
                # Build modules:

                ## Encoder: x -> [e1, e2]
                ## Predictor: e1 -> y
                ## Noisy-transformer: e1 -> noisy_e1
                ## Decoder: [noisy_e1, e2] -> x
                
                self.encoder, self.predictor, self.decoder = \
                    self.module_builder.build_default_modules(
                        ['encoder', 'predictor', 'decoder']
                    )
                
                self.noisy_transformer = self.module_builder.build_module(
                    'noisy_transformer', name='noisy_transformer',
                    build_kwargs={
                        'params': [self.config.dropout_rate]
                    }
                )

                ## Disentanglers:
                self.disentangler1 = self.module_builder.build_module(
                    'disentangler', name='disentangler1',
                    build_kwargs={
                        'input_dim': self.model_config.embedding_dim_1,
                        'output_dim': self.model_config.embedding_dim_2
                    }
                )
                self.disentangler2 = self.module_builder.build_module(
                    'disentangler', name='disentangler2',
                    build_kwargs={
                        'input_dim': self.model_config.embedding_dim_2,
                        'output_dim': self.model_config.embedding_dim_1
                    }
                )
                
                ## z_discriminator:
                if self.config.bias:
                    self.z_discriminator = self.module_builder.build_module(
                        'z_discriminator', name='z_discriminator'
                    )

                # Build 2 copies of the connected network            
                main_model = self.__build_connected_network_train(main=True)
                adv_model = self.__build_connected_network_train(main=False)
                
                models = [main_model, adv_model]
    
            # Parallelize over GPUs
            if self.config.num_gpus > 1:
                for i in range(len(models)):
                    models[i] = \
                        multi_gpu_model(models[i], gpus=self.config.num_gpus)
            
            # Create final model
            
            ## Gather params
            main_params = self.encoder.trainable_weights + \
                self.predictor.trainable_weights + self.decoder.trainable_weights
            adv_params = self.disentangler1.trainable_weights + \
                self.disentangler2.trainable_weights
            if self.config.bias:
                adv_params.extend(self.z_discriminator.trainable_weights)
            
            ## Build keras_adversarial model
            self.model_train = AdversarialModel(
                player_models=models,
                player_params=[main_params, adv_params],
                player_names=['main_model', 'adv_model']
            )
    
    def compile_model(self):
        assert self.model_train is not None, 'run build_model_train()'
        
        optimizers = self.config.optimizers
        losses = self.config.losses
        
        main_loss_weights = [lw for lw in self.config.loss_weights]
        adv_loss_weights = [lw for lw in self.config.loss_weights]
        player_compile_kwargs = [
            {
                'loss_weights': main_loss_weights,
                'metrics': self.config.metrics
            },
            {
                'loss_weights': adv_loss_weights,
                'metrics': self.config.metrics
            }
        ]
        
        adversarial_optimizer = AdversarialOptimizerScheduled(
            [int(p) for p in list(self.config.training_schedule)]
        )
        self.model_train.adversarial_compile(
            adversarial_optimizer=adversarial_optimizer,
            player_optimizers=optimizers,
            loss=losses, player_compile_kwargs=player_compile_kwargs
        )
        
        self.compiled = True
        
    def build_compiled_model(self):
        self.build_model_train()
        self.compile_model()
    
    def fit(self, dtrain, dvalid, streaming_data=False, epochs=None,
            callbacks=None, training_steps=None, validation_steps=None):
        assert self.compiled, 'run compile_model() before training'
        
        if streaming_data:
            train_generator = dtrain
            valid_generator = dvalid
            
            self.model_train.fit_generator(
                train_generator,
                steps_per_epoch=training_steps,
                validation_data=valid_generator,
                validation_steps=validation_steps,
                callbacks=callbacks,
                epochs=epochs
            )
        else:
            xtrain, ytrain = dtrain
            xvalid, yvalid = dvalid

            self.model_train.fit(
                x=xtrain, y=ytrain,
                validation_data=(xvalid, yvalid),
                callbacks=callbacks, epochs=epochs,
                batch_size=(self.config.batch_size * self.config.num_gpus)
            )        
    
    def build_model_inference(self, checkpoint_epoch=None):
        if self.model_inference is None:
            device = '/cpu:0' if self.config.num_gpus > 1 else '/gpu:0'
            with tf.device(device):
                self.encoder = self.module_builder.build_module(
                    'encoder', epoch=checkpoint_epoch
                )
                self.predictor = self.module_builder.build_module(
                    'predictor', epoch=checkpoint_epoch
                )
                
                x = self.encoder.inputs[0]
                e1, _ = self.encoder(x)
                y = self.predictor(e1)
                self.model_inference = Model(x, y)
            
            # Parallelize over GPUs
            if self.config.num_gpus > 1:
                self.model_inference = multi_gpu_model(
                    self.model_inference, gpus=self.config.num_gpus
                )
    
    def predict(self, data, streaming_data=False, prediction_steps=None):
        assert self.model_inference is not None, 'run build_model_inference() first'
        
        if streaming_data:
            return self.model_inference.predict_generator(data, steps=prediction_steps)
        else:
            return self.model_inference.predict(data)
