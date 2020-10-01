# U-Net
# Written by Dr Roushanak Rahmat
########################################################################
import numpy as np
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import matplotlib.pyplot as plt

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_cols = 512
img_rows = 512
epochs = 4001
batch_size = 5

model_dir = '/content/gdrive/My Drive/models/'

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = BatchNormalization()(pool1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = BatchNormalization()(pool2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = BatchNormalization()(pool3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = BatchNormalization()(pool4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2), )(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    model.summary()
    model.compile(optimizer=Adam(lr=0.001, clipvalue=1., clipnorm=1.), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def train():

    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)

    train_data = np.load('/content/gdrive/My Drive/DRIVE/imgs_train.npz')
    imgs_train, imgs_mask_train = train_data['imgs'], train_data['imgs_mask']

    imgs_train = np.delete(imgs_train, -1, 0)
    imgs_mask_train = np.delete(imgs_mask_train, -1, 0)
    imgs_train = imgs_train.astype('float32')
    imgs_train = imgs_train / imgs_train.max()
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train[imgs_mask_train > 0] = 1.0

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = get_unet()

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)

    model_checkpoint = ModelCheckpoint('model4000.h5', monitor='val_loss', save_best_only=True)

    model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, nb_epoch=epochs, verbose=1, shuffle=True,
              callbacks=[model_checkpoint])

    model.save_weights(model_dir + "modelaug{}.h5".format(epochs))


def predict():

    print('- ' * 30)
    print('Get the model structure...')
    print('- ' * 30)

    model = get_unet()

    print('- ' * 30)
    print('Loading and preprocessing test data...')
    print('- ' * 30)

    test_data = np.load('/content/gdrive/My Drive/DRIVE/imgs_test.npz')
    imgs_test, imgs_mask_test_true = test_data['imgs'], test_data['imgs_mask']
    
    print(imgs_test.shape)
    print(imgs_test.dtype)
    print(type(imgs_test))

    # Renormalizing the test masks
    imgs_mask_test_true = imgs_mask_test_true.astype('float32')
    imgs_mask_test_true[imgs_mask_test_true > 0] = 1.0

    imgs_test = imgs_test.astype('float32')
    imgs_test = imgs_test / imgs_test.max()

    print('- ' * 30)
    print('Loading saved weights...')
    print('- ' * 30)
    model.load_weights(model_dir + 'modelaug{}.h5'.format(epochs))

    print('- ' * 30)
    print('Predicting masks on test data...')
    print('- ' * 30)

    imgs_mask_pred = model.predict(imgs_test, verbose=1)

    np.save('imgs_mask_test.npy', imgs_mask_pred)

    dice_all = []

    ind=0
    for impred, im, imtest in zip(imgs_mask_pred[:5], imgs_test[:5], imgs_mask_test_true[:5]):
        dice_all.append(dice_coef(imtest, impred))


        plt.figure()
        plt.title("Ground truth and prediction for test set. Dice {}".format(dice_all[-1]))

        plt.subplot(1, 3, 1)
        plt.title('Image')
        plt.imshow(np.squeeze(im))
        plt.subplot(1, 3, 2)
        plt.title('Ground truth')
        plt.imshow(np.squeeze(imtest))
        plt.subplot(1, 3, 3)
        plt.title('Prediction')
        plt.imshow(np.squeeze(impred))
        plt.savefig('img/test{}_epoch{}.png'.format(ind, epochs))
        ind +=1

if __name__ == '__main__':
    # train()
    predict()