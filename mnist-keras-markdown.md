

```python
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
#from keras import backend as K
import numpy as np
from keras.preprocessing import image
import scipy.misc

```


```python
batch_size = 128
num_classes = 10
epochs = 12
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```


```python
if keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
```


```python
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
```

    x_train shape: (60000, 28, 28, 1)
    60000 train samples
    10000 test samples



```python
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```


```python
print(y_train)
print(y_test)
print(y_train.shape[0])
print(y_train.shape[1])
print(y_test.shape[0])
print(y_test.shape[1])
```

    [[ 0.  0.  0. ...,  0.  0.  0.]
     [ 1.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     ..., 
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  1.  0.]]
    [[ 0.  0.  0. ...,  1.  0.  0.]
     [ 0.  0.  1. ...,  0.  0.  0.]
     [ 0.  1.  0. ...,  0.  0.  0.]
     ..., 
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]]
    60000
    10
    10000
    10



```python
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
```


```python
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
```


```python
model.fit(x_train, y_train,
          batch_size=120,
          epochs=5,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/5
    60000/60000 [==============================] - 210s - loss: 0.3282 - acc: 0.8996 - val_loss: 0.0782 - val_acc: 0.9779
    Epoch 2/5
    60000/60000 [==============================] - 226s - loss: 0.1124 - acc: 0.9669 - val_loss: 0.0538 - val_acc: 0.9834
    Epoch 3/5
    60000/60000 [==============================] - 219s - loss: 0.0836 - acc: 0.9751 - val_loss: 0.0436 - val_acc: 0.9851
    Epoch 4/5
    60000/60000 [==============================] - 218s - loss: 0.0711 - acc: 0.9789 - val_loss: 0.0379 - val_acc: 0.9866
    Epoch 5/5
    60000/60000 [==============================] - 224s - loss: 0.0601 - acc: 0.9822 - val_loss: 0.0354 - val_acc: 0.9888
    Test loss: 0.035373257892
    Test accuracy: 0.9888



```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 26, 26, 32)        320       
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 24, 24, 64)        18496     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 12, 12, 64)        0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 12, 12, 64)        0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 9216)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 128)               1179776   
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 128)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 1,199,882
    Trainable params: 1,199,882
    Non-trainable params: 0
    _________________________________________________________________



```python
#def rgb2gray(rgb):
  #  return np.dot(rgb[...,:3],[0.299,0.587,0.114])

img_path = 'test0.png'
#x= ImageDataGenerator( )
#img= x.flow_from_directory('/',color_mode='grayscale')
img = image.load_img(img_path, target_size=(28,28,),grayscale=True)
scipy.misc.imresize(img, (28, 28))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
result = model.predict(x)

print('Predicted:', result)
```

    Predicted: [[ 0.  0.  0.  0.  0.  0.  0.  0.  1.  0.]]



```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
