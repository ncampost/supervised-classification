
# Tools
import numpy as np
import matplotlib.pyplot as plt

# Data
from keras.datasets import mnist

# Models and optimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras import optimizers
from keras.models import load_model
from keras.utils import to_categorical


def main():
    # Get data.
    # X[i] is a 784-dim vector of [0,1] greyscale values.
    # Y[i] is a scalar indicating digit class. 
    x_train, y_train, x_test, y_test = import_format_data(flatten=True)
    n_train = 60000
    n_predict = 10000

    # Random forest classifier
    # Validation accuracy: 0.966600

    RF = RandomForestClassifier(n_estimators=47, max_features='sqrt', max_depth=130)
    
    # Logistic regression one-versus-many classifier
    # Validation accuracy: 0.9157

    LR = LogisticRegression(C=0.375)

    # Support vector machine one-versus-one classifier
    # Validation accuracy: 0.9446

    SVM = SVC(kernel='linear', C=0.05)

    # Neural network classifier
    # Validation accuracy: 0.9742

    feat_dim = x_train.shape[1]
    n_classes = 10
    train_new = False
    if train_new:
        NN = build_mlp(feat_dim, n_classes)
        NN.compile(optimizer=optimizers.Adadelta(), loss='categorical_crossentropy')

        print "Fitting NN..."
        NN.fit(x_train[:n_train], to_categorical(y_train[:n_train]), epochs=30)
        NN.save('2111_fulltrain.h5')
    else:
        NN = load_model('2111_fulltrain.h5')

    # Convolutional neural network
    # Validation accuracy: 0.9816

    # Get unflattened data
    X_train, _, X_test, _ = import_format_data(flatten=False)

    # Parameters to set up the network and reshape data for what the network expects
    train_new_cnn = False
    retrain_cnn = False
    X_dim = X_train.shape[1]
    Y_dim = X_train.shape[2]
    num_img = X_train.shape[0]
    input_shape = (X_dim, Y_dim, 1)
    X_train = X_train.reshape(num_img, X_dim, Y_dim, 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    if train_new_cnn:
        CNN = build_cnn(n_classes, input_shape)
        print cnn.summary()
        CNN.compile(optimizer=optimizers.Adadelta(), loss='categorical_crossentropy')
        
        print "Fitting CNN..."
        CNN.fit(X_train[:n_train], to_categorical(y_train[:n_train]), epochs=30)
        CNN.save('CNN_new.h5')
    elif retrain_cnn:
        CNN = load_model('CNN.h5')
        CNN.fit(X_train[:n_train], to_categorical(y_train[:n_train]), epochs=10)
        CNN.save('CNN.h5')
    else:
        CNN = load_model('CNN.h5')

    # Joint model
    # Test score: 0.9713

    # Weight each model with its training validation score, then weight predictions after fitting models.
    RF_weight = 0.9666
    LR_weight = 0.9157
    SVM_weight = 0.9446
    NN_weight = 0.9742
    CNN_weight = 0.9816

    print "Fitting RF..."
    RF.fit(x_train, y_train)
    print "Fitting LR..."
    LR.fit(x_train, y_train)
    print "Fitting SVM..."
    SVM.fit(x_train, y_train)
    
    print "Predicting RF..."
    RF_preds_single = RF.predict(x_test)
    RF_preds = np.zeros((n_predict, n_classes))
    RF_preds[np.arange(n_predict), RF_preds_single] = RF_weight
    RF_acc = np.sum(np.equal(RF_preds_single, y_test)) / float(n_predict)

    print "Predicting LR..."
    LR_preds_single = LR.predict(x_test)
    LR_preds = np.zeros((n_predict, n_classes))
    LR_preds[np.arange(n_predict), LR_preds_single] = LR_weight
    LR_acc = np.sum(np.equal(LR_preds_single, y_test)) / float(n_predict)

    print "Predicting SVM..."
    SVM_preds_single = SVM.predict(x_test)
    SVM_preds = np.zeros((n_predict, n_classes))
    SVM_preds[np.arange(n_predict), SVM_preds_single] = SVM_weight
    SVM_acc = np.sum(np.equal(SVM_preds_single, y_test)) / float(n_predict)

    print "Predicting NN..."
    NN_preds = NN.predict(x_test) * NN_weight
    NN_preds_single = np.argmax(NN_preds,axis=1)
    NN_acc = np.sum(np.equal(NN_preds_single, y_test)) / float(n_predict)

    print "Predicting CNN..."
    CNN_preds = CNN.predict(X_test) * CNN_weight
    CNN_preds_single = np.argmax(CNN_preds,axis=1)
    CNN_acc = np.sum(np.equal(CNN_preds_single, y_test)) / float(n_predict)

    print "Gathering predictions and verifying..."
    preds = np.argmax(RF_preds + LR_preds + SVM_preds + NN_preds + CNN_preds, axis=1)
    joint_acc = np.sum(np.equal(preds, y_test)) / float(n_predict)
    
    print "RF val %f" % (RF_acc)
    print "LR val %f" % (LR_acc)
    print "SVM val %f" % (SVM_acc)
    print "NN val %f" % (NN_acc)
    print "CNN val %f" % (CNN_acc)
    print "Joint val %f" % (joint_acc)
    

    # Example gridsearch over random forest classifier. 
    
    #cand_hyperparameters = [{'n_estimators': [43, 44, 45, 46, 47], 
    #                            'max_depth': [125, 130, 135, 140],
    #                            'max_features': ['auto', 'sqrt'],
    #                        }]

    #gridsearch_model(RandomForestClassifier, cand_hyperparameters, x_train, y_train, n_train)
    



# Perform a hyperparameter gridsearch 
def gridsearch_model(model, cand_hyperparameters, x_train, y_train, n_train):
    RF = model()
    RF_params = GridSearchCV(RF, cand_hyperparameters, verbose=2)
    RF_params.fit(x_train[:n_train], y_train[:n_train])
    print "Best params:"
    print RF_params.best_params_

# Build a MLP neural net.
def build_mlp(feat_dim, n_classes):
    input_shape = (feat_dim,)
    inputs = Input(input_shape)
    layer = Dense(2 * feat_dim, activation='tanh')(inputs)
    layer = Dense(1 * feat_dim, activation='tanh')(layer)
    layer = Dense(1 * feat_dim, activation='tanh')(layer)
    layer = Dense(1 * feat_dim, activation='tanh')(layer)
    out = Dense(n_classes, activation='softmax')(layer)

    return Model(inputs, out)

# Build a conv. neural net.
def build_cnn(n_classes, input_shape):
    cnn = Sequential()
    cnn.add(Conv2D(32, kernel_size=(8, 8), strides=(2, 2),
                    activation='relu',
                    input_shape=input_shape))
    cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    cnn.add(Conv2D(64, (4, 4), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(64, activation='relu'))
    cnn.add(Dense(n_classes, activation='softmax'))

# Thanks to shchao for writing this histogram code!
def make_histogram():
    # create bar plot of rmse for different methods/features used
    # data to plot
    n_groups = 6
    acc_rf = 0.9674
    acc_lr = 0.9201
    acc_svm = 0.9466
    acc_nn = 0.9807
    acc_cnn = 0.9922
    acc_joint = 0.9789
    
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.4
    opacity = 0.8
    
    rects1 = plt.bar(0, acc_rf, bar_width,
                    alpha=opacity,
                    color='b')
    
    rects2 = plt.bar(1, acc_lr, bar_width,
                    alpha=opacity,
                    color='c')

    rects3 = plt.bar(2, acc_svm, bar_width,
                    alpha=opacity,
                    color='g')

    rects4 = plt.bar(3, acc_nn, bar_width,
                    alpha=opacity,
                    color='y')

    rects5 = plt.bar(4, acc_cnn, bar_width,
                    alpha=opacity,
                    color='r')
    
    rects6 = plt.bar(5, acc_joint, bar_width,
                    alpha=opacity,
                    color='k')

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off')         # ticks along the top edge are off
        
    #plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Tuned Model Performance')
    plt.xticks(index, ('Random Forest', 'Logistic Regression', 'Support Vector Machine', 'Neural Net', 'Conv Neural Net', 'Joint Model'))

    plt.axhline(y=0.8832, linewidth=2, linestyle='--', color='m', label='Logistic Regression benchmark')
    plt.axhline(y=0.9098, linewidth=2, linestyle='--', color='r', label='Random Forest benchmark')
    plt.axhline(y=1, linewidth=2,  linestyle='--', color='k', label='100% line')

    # Shrink current axis's height by 20% on the bottom
    box = ax.get_position()
    #ax.set_position([box.x0, box.y0 + box.height,
    #                 box.width, box.height])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
            fancybox=True, shadow=True, title='Regression Technique', ncol=2)

    plt.legend()
    plt.tight_layout()
    #plt.savefig("model_hist.png")
    plt.show()

def import_format_data(flatten):
    # Get dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Turn [0,255] values to [0,1] values
    x_train = x_train.astype('float64') / 255.0
    x_test = x_test.astype('float64') / 255.0
    if flatten:
        # Turn (N, A, B) array into (N, A*B) flattened arrays
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    main()
