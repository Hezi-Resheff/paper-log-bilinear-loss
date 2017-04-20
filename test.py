"""
Put it all together with a simple MNIST exmaple
"""

from tensorflow.examples.tutorials.mnist import input_data
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix

from models import mnist_model
from loss import bilinear_loss
from util import *

DATA_DIR = ""

LRATE = 5e-4                     # Learning rate for the model
EPOCHS = 10                      # How many epochs to train for
BATCH_SIZE = 50                  #
VERBOSITY = 1                    #
N_SPOTS = 10                     # Number of spots in the mask of "bad errors"
ALPHA = .9                       # Trade-off parameter. Higher value puts more weight on not making errors in the mask


mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
data = {
    "X_train": mnist.train.images.reshape(-1, 28, 28, 1),
    "X_valid": mnist.validation.images.reshape(-1, 28, 28, 1),
    "X_test": mnist.test.images.reshape(-1, 28, 28, 1),
    "Y_train": mnist.train.labels,
    "Y_valid": mnist.validation.labels,
    "Y_test": mnist.test.labels
}


# generate a random matrix with locations we don't want to make mistakes in
cm = make_random_spots_cm(N_SPOTS, normalize=True)

# generate a loss function (bilinar+cross-entropy) to reflect the random spots
loss = bilinear_loss(cm, alpha=.9)

# train the model
model = mnist_model()
model.summary()
model.compile(loss=loss, optimizer=Adam(LRATE), metrics=['accuracy'])
model.fit(data["X_train"], data["Y_train"],
          nb_epoch=EPOCHS,
          validation_data=(data["X_valid"], data["Y_valid"]),
          verbose=VERBOSITY,
          callbacks=None)

# What percent of all errors is in the mask?
mask = cm > 0
pred = model.predict(data["X_test"])
model_cm = confusion_matrix(data["Y_test"].argmax(axis=1), pred.argmax(axis=1))
model_cm_norm = confusion_matrix_normalizer(model_cm, strip_diagonal=True, normalize_rows=False, normalize_matrix=True)
percent_error_in_mask = model_cm_norm[mask].sum() * 100.
print("The percent of all error in the mask is: {:.4f}%".format(percent_error_in_mask))
