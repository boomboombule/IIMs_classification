from keras.applications.vgg19  import VGG19
from keras.applications.vgg16  import VGG16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,concatenate
from tensorflow.keras.optimizers import Adam
import sys

def ModelVGG16():
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)

    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in model.layers:
        layer.trainable = True

    opt = Adam(learning_rate=0.00002)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
def ModelVGG19():
    base_model = VGG19(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(4, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in model.layers:
        layer.trainable = True
    # opt = Adam(learning_rate=0.00002)
    opt = Adam()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
def ConcatedModel():
    vg_model_16 = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    vg_model_19 = VGG19(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    x16 = vg_model_16.output
    x16 = GlobalAveragePooling2D()(x16)

    x19 = vg_model_19.output
    x19 = GlobalAveragePooling2D()(x19)

    addLayer = concatenate([x16, x19])

    x = Dense(1024, activation='relu')(addLayer)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    output = Dense(4, activation='softmax')(x)

    for layer in vg_model_16.layers:
        layer.trainable = True
    for layer in vg_model_19.layers:
        layer.trainable = True

    for layer in vg_model_19.layers:
        layer._name = layer._name + str('_C')

    vg_model = Model(inputs=[vg_model_16.input, vg_model_19.input], outputs=output)
    #opt = Adam(learning_rate=0.00002)
    opt = Adam()
    vg_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return vg_model






















