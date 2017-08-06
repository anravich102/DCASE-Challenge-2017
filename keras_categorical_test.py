from keras.utils import to_categorical

import numpy as np 

y = np.hstack( (np.zeros(10), np.ones(10)) ) .reshape(20,1)

y = to_categorical(y, 2)

print(y)