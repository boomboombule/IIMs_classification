from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, ZeroPadding2D, Dense, Dropout, \
    Activation, Flatten, BatchNormalization, Input,add,GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.tools.depthwise_conv2d import DepthwiseConv2D1
from models.tools.cbam_module import cbam_module

def conv_block(layer, filters, kernel_size=(3, 3), strides=(1, 1), padding='valid', l2_reg=0.0, name=None):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding=padding,
               kernel_regularizer=l2(l2_reg),
               kernel_initializer="he_normal",
               name=name)(layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def DBFF_CNN(input_shape, num_classes, l2_reg=0.0, weights=None):
    # Group1
    input_0 = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(2, 2))(input_0 )
    x = conv_block(x, filters=48, kernel_size=(11, 11),strides=(4, 4), padding="valid", l2_reg=l2_reg, name='Conv_1_96_11x11_4')
    x = DepthwiseConv2D1(48, (3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid", name="maxpool_1_3x3_2")(x)
    x = cbam_module(x)
    x = BatchNormalization()(x)
    x1 = Activation('relu')(x)
    # Group2
    # Group2.1
    x = conv_block(x1, filters=96, kernel_size=(5, 5),strides=(1, 1), padding="same", l2_reg=l2_reg, name="Conv_2_1.1_256_5x5_1")
    x = cbam_module(x)
    x = DepthwiseConv2D1(96, (3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid", name="maxpool_2_2_1_3x3_2")(x)
    x2_1 = BatchNormalization()(x)
    # Group2.2
    x = conv_block(x1, filters=96, kernel_size=(5, 5),strides=(1, 1), padding="same", l2_reg=l2_reg, name="Conv_2_2.1_256_5x5_1")
    x = cbam_module(x)
    x = DepthwiseConv2D1(96, (3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = conv_block(x, filters=96, kernel_size=(5, 5),strides=(1, 1), padding="same", l2_reg=l2_reg, name="Conv_2_2.2_256_5x5_1")
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid", name="maxpool_2_2_3x3_2")(x)
    x2_2 = BatchNormalization()(x)
    
    DBFF1 = add ([x2_1,x2_2])
    DBFF1 = Activation('relu')(DBFF1)

    # Group3
    # Group3.1
    x = conv_block(DBFF1, filters=256, kernel_size=(3, 3),strides=(1, 1), padding="same", l2_reg=l2_reg, name="Conv_3_2.1_384_3x3_2")
    x = DepthwiseConv2D1(256, (3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x3_1 = conv_block(x, filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", l2_reg=l2_reg, name="Conv_3_2.2_384_3x3_3")
    # Group3.2
    x = conv_block(DBFF1, filters=256, kernel_size=(3, 3),strides=(1, 1), padding="same", l2_reg=l2_reg, name="Conv_3_1_384_3x3_1")
    x = DepthwiseConv2D1(256, (3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x3_2 = Activation('relu')(x)
   
    DBFF2 = add ([x3_1,x3_2])
    DBFF2 = Activation('relu')(DBFF2)

    # Group4
    # Group4.1
    x = conv_block(DBFF2, filters=384, kernel_size=(3, 3),strides=(1, 1), padding="same", l2_reg=l2_reg, name="Conv_4_2.1_384_3x3_1")
    x = conv_block(x, filters=384, kernel_size=(3, 3),strides=(1, 1), padding="same", l2_reg=l2_reg, name="Conv_4_2.2_384_3x3_1")
    x = DepthwiseConv2D1(384, (3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x4_1 = Activation('relu')(x)
    # Group4.2
    x = conv_block(DBFF2, filters=384, kernel_size=(3, 3),strides=(1, 1), padding="same", l2_reg=l2_reg, name="Conv_4_1.1_384_3x3_1")
    x = DepthwiseConv2D1(384, (3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = conv_block(x, filters=384, kernel_size=(3, 3),strides=(1, 1), padding="same", l2_reg=l2_reg, name="Conv_4_1.2_384_3x3_1")
    x = DepthwiseConv2D1(384, (3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x4_2 = Activation('relu')(x)
    
    DBFF3 = add ([x4_1,x4_2])
    DBFF3 = Activation('relu')(DBFF3)

    # Group5
    x = conv_block(DBFF3, filters=256, kernel_size=(3, 3),strides=(1, 1), padding="same", l2_reg=l2_reg, name="Conv_5_256_3x3_1")
    x = DepthwiseConv2D1(256, (3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid", name="maxpool_3_3x3_2")(x)

    # GAP
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # FC
    x = Dense(units=256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # SOFTMAX
    x = Dense(units=num_classes)(x)
    x = BatchNormalization()(x)
    x = Activation("softmax")(x)
    
    model = Model(inputs=input_0, outputs=x)
    return model