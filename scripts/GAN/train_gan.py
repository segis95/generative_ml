import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

import hydra
import wandb
from wandb import Image as wandb_Image
from wandb import Table
from omegaconf import OmegaConf

def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)

    images_mnist = np.concatenate([x_train, x_test], axis=0).reshape((-1, 28, 28, 1)) / 255.0
    return images_mnist

def create_discriminator():
    discriminator = keras.Sequential(
        [
            keras.layers.InputLayer((28, 28, 1)),
            keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Flatten(),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ],
        name="discriminator",
    )
    
    return discriminator
    
def create_generator(generator_hidden_size=100):
    generator = keras.Sequential(
        [
            keras.layers.InputLayer((generator_hidden_size,)),
            keras.layers.Dense(7 * 7 * generator_hidden_size),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Reshape((7, 7, generator_hidden_size)),
            keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
        ],
        name="generator",
    )
    
    return generator

@keras.saving.register_keras_serializable()
class GAN(keras.Model):
    def __init__(self, discriminator, generator, wandb_logging=True):
        super().__init__()
        self.wandb_logging = wandb_logging
        self.generator_hidden_size = generator.input.shape[1]
        self.e2e_inp = keras.layers.Input((self.generator_hidden_size,))
        
        self.discriminator = discriminator
        self.generator = generator
        self.e2e_model = keras.Model(inputs=self.e2e_inp, outputs=self.discriminator(self.generator(self.e2e_inp)))

    def compile(self, disc_learning_rate, gen_learning_rate):
        super().compile()
        self.disc_optimizer = keras.optimizers.Adam(learning_rate=disc_learning_rate)
        self.e2e_optimizer = keras.optimizers.Adam(learning_rate=gen_learning_rate)
        
        self.loss_e2e = keras.losses.BinaryCrossentropy()
        self.loss_disc = keras.losses.BinaryCrossentropy()

        self.acc_e2e = keras.metrics.Accuracy()
        self.acc_disc = keras.metrics.Accuracy()

        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")


    @property
    def metrics(self):
        return [self.acc_e2e, self.acc_disc, self.disc_loss_tracker, self.gen_loss_tracker]
        
    def train_step(self, data):
        
        img_real = data
        batch_size = tf.shape(img_real)[0]

        eta = tf.random.normal(shape=(batch_size, self.generator_hidden_size), mean=0.0, stddev=1.0, dtype=tf.float32)
        img_gen = self.generator(eta)
        data_mixed = tf.concat([img_real, img_gen], axis=0)
        data_mixed_y = tf.concat([
                                    tf.ones((batch_size, )),
                                    tf.zeros((batch_size, ))
                                ], axis=0)

        
        with tf.GradientTape() as tape1:
            disc_pred = self.discriminator(data_mixed)
            disc_loss = self.loss_disc(data_mixed_y, disc_pred)

        grad_disc = tape1.gradient(disc_loss, self.discriminator.trainable_weights)
        self.disc_optimizer.apply_gradients(zip(grad_disc, self.discriminator.trainable_weights))
        self.acc_disc.update_state(data_mixed_y, disc_pred)


        eta = tf.random.normal(shape=(2 * batch_size, self.generator_hidden_size), mean=0.0, stddev=1.0, dtype=tf.float32)
        data_y = tf.ones((2 * batch_size, ))
        with tf.GradientTape() as tape2:
            e2e_pred = self.e2e_model(eta)
            e2e_loss = self.loss_e2e(data_y, e2e_pred)

        grad_gen = tape2.gradient(e2e_loss, self.generator.trainable_weights)
        self.e2e_optimizer.apply_gradients(zip(grad_gen, self.generator.trainable_weights))
        self.acc_e2e.update_state(data_y, e2e_pred)


        self.gen_loss_tracker.update_state(e2e_loss)
        self.disc_loss_tracker.update_state(disc_loss)

        return {"disc_acc":self.acc_disc.result(), "gen_acc":self.acc_e2e.result(),   "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result()}

    def get_config(self):
        base_config = super().get_config()
        config = {
            "discriminator": keras.saving.serialize_keras_object(self.discriminator),
            "generator": keras.saving.serialize_keras_object(self.generator),
            "wandb_logging": self.wandb_logging
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        discriminator_config = config.pop("discriminator")
        discriminator = keras.saving.deserialize_keras_object(discriminator_config)

        generator_config = config.pop("generator")
        generator = keras.saving.deserialize_keras_object(generator_config)

        return cls(discriminator, generator, **config)

class GeneratorProgressionCallback(keras.callbacks.Callback):
    def __init__(self, generate_images_per_epoch):
        super().__init__()
        self.generate_images_per_epoch = generate_images_per_epoch
        self.table_data = []

    @staticmethod
    def source_data_generator(generator_input_shape, images_per_epoch):
        eta = np.random.normal(loc=0.0, scale=1.0, size=(images_per_epoch, generator_input_shape))
        labels = np.array([1] * images_per_epoch)
        
        return eta, labels

    def on_train_batch_end(self, batch, logs=None):
        if self.model.wandb_logging:
            for metric, value in logs.items():
                wandb.log({metric: value})
    
    def on_epoch_end(self, epoch, logs=None):
        X, y = self.source_data_generator(self.model.generator.input.shape[1], self.generate_images_per_epoch)
        generated = self.model.generator.predict(X)

        fig, axes = plt.subplots(nrows=1, ncols=self.generate_images_per_epoch, figsize=(self.generate_images_per_epoch, 2))
        plt.axis('off')
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        
        for elid in range(self.generate_images_per_epoch):
            axes[elid].imshow(generated[elid], cmap='gray')

        self.table_data.append([epoch, wandb_Image(fig)])        

    def on_train_end(self, logs=None):
        table = Table(data=self.table_data, columns=["epoch", "examples_of_generation"])
        wandb.log({'Generation examples from different training epochs': table})

@hydra.main(version_base=None, config_path="../../configs", config_name="train_gan_config")
def main(config):
    
    if config.logging.wandb.enabled:
        wandb.config = OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True
        )

        wandb.init(project=config.logging.wandb.project)

    discriminator = create_discriminator()
    generator = create_generator(generator_hidden_size=config.model.hidden_space_size)
    gan = GAN(discriminator, generator, wandb_logging=config.logging.wandb.enabled)
    gan.compile(config.training.discriminator_learning_rate, config.training.generator_learning_rate)
    images_mnist = load_data()
    callback = []
    gan.fit(images_mnist, epochs=config.training.epochs,
            batch_size=config.training.batch_size,
            callbacks=[GeneratorProgressionCallback(config.logging.wandb.n_examples_per_epoch)])

    if config.logging.save_weights:
        gan.save("gan.keras")



if __name__ == '__main__':
    main()