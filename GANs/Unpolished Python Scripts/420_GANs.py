import pydotplus
import os, shutil
import matplotlib.pyplot as plt
import random
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
from numpy.random import rand, normal, randn, randint
from numpy import zeros, vstack, asarray, expand_dims, ones
from keras.models import Sequential
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Dense, Reshape, BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU, ReLU
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator 

random.seed(10)

filelist = glob.glob(r'C:\Users\Harshinee Sriram\OneDrive\Desktop\UBC STUDY\CPSC 540\Project\DATASETS\Working\Train\music\*.png')
trainX = np.array([np.array(Image.open(this_img)) for this_img in filelist])

#filelist2 = glob.glob(r'C:\Users\Harshinee Sriram\OneDrive\Desktop\UBC STUDY\CPSC 540\Project\DATASETS\Working\Train\not_music\*.png')
#trainX2 = np.array([np.array(Image.open(this_img)) for this_img in filelist2])

#trainX2 = trainX2[:, :, :, 0]
#trainX = np.vstack((trainX, trainX2))
print("Size of Training X:", trainX.shape)

filelist = glob.glob(r'C:\Users\Harshinee Sriram\OneDrive\Desktop\UBC STUDY\CPSC 540\Project\DATASETS\Working\Validation\music\*.png')
testX = np.array([np.array(Image.open(this_img)) for this_img in filelist])

#filelist2 = glob.glob(r'C:\Users\Harshinee Sriram\OneDrive\Desktop\UBC STUDY\CPSC 540\Project\DATASETS\Working\Validation\not_music\*.png')
#testX2 = np.array([np.array(Image.open(this_img)) for this_img in filelist2])

#testX2 = testX2[:, :, :, 0]
#testX = np.vstack((testX, testX2))
print("Size of Testing X:", testX.shape)

trainY = []
for i in range(1214):
    trainY.append('1')
    
#for i in range(1214):
#    trainY.append('1')

trainY = np.array(trainY)
print("Size of Training Y:", trainY.shape)

testY = []
for i in range(519):
    testY.append('1')
    
#for i in range(519):
#    testY.append('1')

testY = np.array(testY)
print("Size of Testing Y:", testY.shape)

# Defining the discriminator for this GAN

def define_discriminator():
    
    model = Sequential()
    model.add(Conv2D(128, (3,3), strides=(1, 1), padding='same', input_shape=(106, 106, 1), activation='tanh'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(1024, activation='tanh'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

discriminator = define_discriminator()
discriminator.summary()
plot_model(discriminator, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)

def train_preprocessing():
    filelist = glob.glob(r'C:\Users\Harshinee Sriram\OneDrive\Desktop\UBC STUDY\CPSC 540\Project\DATASETS\Working\Train\music\*.png')
    trainX = np.array([np.array(Image.open(this_img)) for this_img in filelist])
    trainX = expand_dims(trainX, axis=-1)
    trainX = trainX.astype('float32')
    trainX = trainX/255.0
    return trainX

def generate_real_samples(dataset, n_samples):
	indices = randint(0, dataset.shape[0], n_samples)
	this_X = dataset[indices]
	this_Y = ones((n_samples, 1))
	return this_X, this_Y

def generate_fake_samples(n_samples):
	X = rand(106 * 106 * n_samples)
	X = X.reshape((n_samples, 106, 106, 1))
	y = zeros((n_samples, 1))
	return X, y

def train_discriminator(model, dataset, n_iter=10, n_batch=64):
    half_batch = int(n_batch / 2)
    for i in range(n_iter):
        X_real, y_real = generate_real_samples(dataset, half_batch)
        _, real_acc = model.train_on_batch(X_real, y_real)
        X_fake, y_fake = generate_fake_samples(half_batch)
        _, fake_acc = model.train_on_batch(X_fake, y_fake)

        print('>%d real=%.0f%% fake=%.0f%%' % (i+1, real_acc*100, fake_acc*100))

number_of_samples_here = 25
dataset = train_preprocessing()
x_real, y_real = generate_real_samples(dataset, number_of_samples_here)
x_real = x_real.reshape((number_of_samples_here, 106, 106))
plt.figure(figsize=(50,50))
for i in range(20):
    plt.subplot(5, 5, 1 + i)
    plt.imshow(x_real[i], cmap='gray')
plt.show()

discriminator = define_discriminator()
train_discriminator(discriminator, dataset)

def define_generator():
    model = Sequential()
    n_nodes = 64 * 53 * 53
    model.add(Dense(n_nodes, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((53, 53, 64)))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1, (53,53), activation='sigmoid', padding='same'))
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

generator = define_generator()
generator.summary()
plot_model(generator, to_file='generator_plot.png', show_shapes=True, show_layer_names=True)

def generate_latent_points(n_samples):
    x_input = randn(100 * n_samples)
    x_input = x_input.reshape(n_samples, 100)
    return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples_2(generator, n_samples):
    x_input = generate_latent_points(n_samples)
    X = generator.predict(x_input)
    y = zeros((n_samples, 1))
    return X, y

n_samples = 10
X, _ = generate_fake_samples_2(generator, n_samples)
plt.figure(figsize=(150,150))
for i in range(n_samples):
	# define subplot
	plt.subplot(5, 5, 1 + i)
	# turn off axis labels
	plt.axis('off')
	# plot single image
	plt.imshow(X[i, :, :, 0], cmap='gray_r')
# show the figure
plt.show()

def GAN(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002))
    return model

gan_model = GAN(generator, discriminator)
gan_model.summary()
plot_model(gan_model, to_file='gan_plot.png', show_shapes=True, show_layer_names=True)

def save_plot(examples, epoch, n=10):
    for i in range(n * n):
        pyplot.subplot(n, n, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
    filename = 'generated_plot_e%03d.png' % (epoch+1)
    pyplot.savefig(filename)
    pyplot.close()
    
def performance(epoch, generator, discriminator, dataset, n_samples=50):
    X_real, y_real = generate_real_samples(dataset, n_samples)
    _, acc_real = discriminator.evaluate(X_real, y_real, verbose=0)
    x_fake, y_fake = generate_fake_samples_2(generator, n_samples)
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    save_plot(x_fake, epoch)
    filename = 'generator_model_%03d.h5' % (epoch + 1)
    generator.save(filename)

def train_GAN(generator, discriminator, gan_model, dataset, n_epochs=10, n_batch=32):
    batch_per_epoch = int(dataset.shape[0]/n_batch)
    half_batch = int(n_batch/2)
    for i in range(n_epochs):
        
        for j in range(batch_per_epoch):
            X_real, y_real = generate_real_samples(dataset, half_batch)
            X_fake, y_fake = generate_fake_samples_2(generator, half_batch)
            X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
            discriminator_loss, _ = discriminator.train_on_batch(X, y)
            X_gan = generate_latent_points(n_batch)
            y_gan = ones((n_batch, 1))
            print(X.shape, y.shape, X_gan.shape, y_gan.shape)
            generator_loss = generator.train_on_batch(X_gan, y_gan)
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, batch_per_epoch, discriminator_loss, generator_loss))
        
        if (i+1) % 10 == 0:
            performance(i, generator, discriminator, dataset)

generator = define_generator()
gan_model = GAN(generator, discriminator)
train_GAN(generator, discriminator, gan_model, dataset)