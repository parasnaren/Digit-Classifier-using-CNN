import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

y_train = train['label']
X_train = train.drop('label', axis=1)
del train

y_train.value_counts()
sns.countplot(y_train)

X_train.isnull().any().describe()
y_train.isnull().any().describe()

# Illumination differences
X_train = X_train / 255.0
test = test / 255.0

# Reshape into 3D matrix
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

# Convert y_train into categoriacal : 2 -> [0,0,1,0,0,0...]
y_train = to_categorical(y_train, num_classes = 10)

random_seed = 2

X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, 
                                                  test_size = 0.1, random_state=random_seed)

g = plt.imshow(X_train[0][:,:,0])

# CNN model

model = Sequential()

model.add(Conv2D(filters=32, kernel_size= (5,5), padding='Same', 
                 activation='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters=32, kernel_size= (5,5), padding='Same', 
                 activation='relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size= (3,3), padding='Same', 
                 activation='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters=64, kernel_size= (3,3), padding='Same', 
                 activation='relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

####### Augmenting the data ##########
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)

##################################

epochs = 10
batch_size = 86

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 
                    validation_data = (X_val, Y_val))

##### Results
results = model.predict(test)
results = np.argmax(results, axis=1)

tmp = pd.read_csv('sample_submission.csv')
tmp['Label'] = results

tmp.to_csv('prediction.csv', index=False)
###################################################################

from PIL import ImageFilter, Image

def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    print(tva)
    return tva

x=[imageprepare('./image1.png')]#file path here
print(len(x))# mnist IMAGES are 28x28=784 pixels
print(x[0])
#Now we convert 784 sized 1d array to 24x24 sized 2d array so that we can visualize it
newArr=[[0 for d in range(28)] for y in range(28)]
k = 0
for i in range(28):
    for j in range(28):
        newArr[i][j]=x[0][k]
        k=k+1

plt.imshow(newArr, interpolation='nearest')
plt.savefig('MNIST_IMAGE.png')
plt.show()

######## Predict the image
temp = np.array(newArr).reshape(-1,28,28,1)
temp = model.predict(temp)
print("Number is :", np.argmax(temp, axis=1))
