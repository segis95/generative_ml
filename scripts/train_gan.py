import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

import wandb
import hydra
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

class GAN(keras.Model):
    def __init__(self, discriminator, generator, hidden_size=100):
        super().__init__()
        self.generator_hidden_size = hidden_size
        self.e2e_inp = keras.layers.Input((hidden_size,))
        
        self.discriminator = discriminator
        self.generator = generator
        self.e2e_model = keras.Model(inputs=self.e2e_inp, outputs=self.discriminator(self.generator(self.e2e_inp)))

    def compile(self):
        super().compile()
        self.disc_optimizer = keras.optimizers.Adam(learning_rate=0.00003)
        self.e2e_optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        
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

def source_data_generator(generator, n_items=100):
    eta = np.random.normal(loc=0.0, scale=1.0, size=(n_items, generator.input.shape[1]))
    labels = np.array([1] * n_items)
    
    return eta, labels
    
@hydra.main(version_base=None, config_path="../configs", config_name="train_gan_config")
def main(config):
    
    if config.wandb.enabled:
        wandb.config = omegaconf.OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True
        )

        wandb.init(project=config.wandb.project)

    discriminator = create_discriminator()
    generator = create_generator()
    gan = GAN(discriminator, generator)
    gan.compile()
    images_mnist = load_data()
    gan.fit(images_mnist, epochs=config.training.epochs, batch_size=config.training.batch_size)

    X, y = source_data_generator(generator, n_items=100)
    generated = generator.predict(X)
    
    fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(30, 15))
    for elid in range(100):
        ax[elid // 10, elid % 10].imshow(generated[elid], cmap='gray')
    
    plt.savefig("gen_.png")

if __name__ == '__main__':
    main()