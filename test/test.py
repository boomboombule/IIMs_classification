
import numpy as np 
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from models.DBFF_CNN import DBFF_CNN
input_shape = (224, 224, 3)
num_classes = 4 # The value is determined by the  classification scene.
testset_dir = './data/S4/k5/test' #Select the weight path of the classification model according to different classification scenarios.
weight_path = './weights/DBFF-CNN/DBFF_S4_k5.h5'
batch_size = 643
model =DBFF_CNN((224,224,3), 4, l2_reg=0.0, weights=None)
model.load_weights(weight_path)

# Prediction on test set
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    testset_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical', 
    shuffle=False)

for i in range(len(test_generator)):
    x_test, y_test = test_generator.__getitem__(i)
    test_true = np.argmax(y_test, axis=1)
    test_pred = np.argmax(model.predict(x_test), axis=1)
    # print(test_true)
    # print(test_pred)
    dataframe = pd.DataFrame({'true_labels':test_true, 'pred_labels':test_pred}, columns=['true_labels', 'pred_labels'])
    if i == 0:
        dataframe.to_csv('./test/test_result/DBFF-CNN/DBFF_S4k5.csv', sep=',', mode='w', index=False)#Save test results
    else:
        dataframe.to_csv('./test/test_result/DBFF-CNN/DBFF_S4k5.csv', sep=',', mode='a', index=False, header=False)

    confusion_m = confusion_matrix(test_true, test_pred)
    df_confusion_m = pd.DataFrame(confusion_m, columns=['0', '1','2','4'], index=['0', '1','2','4'])
    #df_confusion_m = pd.DataFrame(confusion_m, columns=['0', '1'], index=['0', '1'])
    df_confusion_m.index.name = 'Real'
    df_confusion_m.columns.name = 'Predict'

    # print(df_confusion_m)
    print(metrics.confusion_matrix(y_true =test_true,y_pred=test_pred))
    


