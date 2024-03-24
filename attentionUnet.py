from unet_model import conv_block, encoder_block 
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout,Average,Subtract,Multiply,Lambda,Add
from keras.optimizers import Adam,SGD,RMSprop
from keras.utils import plot_model
from keras import backend as K     
from keras.layers import Activation, Reshape, Permute, multiply

def attention_gate(input, gating_signal, num_filters):
    gating_conv = Conv2D(num_filters, (1, 1), padding='same')(gating_signal)
    gating_conv = BatchNormalization()(gating_conv)
    gating_conv = Activation('relu')(gating_conv)

    input_conv = Conv2D(num_filters, (1, 1), padding='same')(input)
    input_conv = BatchNormalization()(input_conv)
    input_conv = Activation('relu')(input_conv)

    add = Add()([input_conv, gating_conv])
    add = Activation('relu')(add)

    attention_coefficients = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(add)

    output = multiply([input, attention_coefficients])
    return output

def decoder_block(input, concat_tensor, num_filters, upconv=True, droprate=0.25, growth_factor=2):
    num_filters //= growth_factor
    if upconv:
        input = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input)
    attention_weighted_concat = attention_gate(concat_tensor, input, num_filters)
    if upconv:
        x = concatenate([input, attention_weighted_concat])
    else:
        x = concatenate([UpSampling2D((2, 2))(input), attention_weighted_concat])
    x = BatchNormalization()(x)
    x = conv_block(x, num_filters)
    x = Dropout(droprate)(x)
    return x, num_filters



def unet_model(n_classes=4, im_sz=32, n_channels=3, n_filters_start=32, growth_factor=2, upconv=True,
               class_weights=[0.2, 0.2, 0.5, 0.1]):
    droprate=0.25
    input_lr = Input((im_sz, im_sz, n_channels))
    input_hr = Input((im_sz, im_sz, n_channels))
    input_near = Input((im_sz, im_sz, n_channels))
    input_far = Input((im_sz, im_sz, n_channels))

    # encoder
    # net for LR
    conv1,pool1,n_filters = encoder_block(input_lr, n_filters_start,droprate=droprate,dropout=False)
    print(n_filters)
    conv2,pool2,n_filters = encoder_block(pool1, n_filters,droprate=droprate)
    print(n_filters)
    conv3,pool3,n_filters = encoder_block(pool2, n_filters,droprate=droprate)
    print(n_filters)
    conv4,pool4,n_filters = encoder_block(pool3, n_filters,droprate=droprate)
    print(n_filters)
    conv5,pool5,n_filters = encoder_block(pool4, n_filters,droprate=droprate)
    print(n_filters)
    feature_lr=conv_block(pool5,n_filters)
    # net for HR
    convs1,pools1,n_filters = encoder_block(input_hr, n_filters_start,droprate=droprate,dropout=False)
    convs2,pools2,n_filters = encoder_block(pools1, n_filters,droprate=droprate)
    convs3,pools3,n_filters = encoder_block(pools2, n_filters,droprate=droprate)
    convs4,pools4,n_filters = encoder_block(pools3, n_filters,droprate=droprate)
    convs5,pools5,n_filters = encoder_block(pools4, n_filters,droprate=droprate)
    feature_hr=conv_block(pools5,n_filters)

    #submodel
    submodel=Model(inputs=input_lr,outputs=feature_lr)
    feature_lr_n=submodel(input_near)
    feature_lr_f=submodel(input_far)
    print(feature_lr_n.shape)
    print(feature_lr_f.shape)   
    dif_near=Subtract()([feature_lr_n, feature_lr])
    dif_far=Subtract()([feature_lr_f, feature_lr])
    
    
    far=Lambda(lambda dif_far: K.abs(1/(1+dif_far)))(dif_far)
    near=Lambda(lambda dif_near: K.abs(dif_near))(dif_near)   

    mul=Multiply()([far,near]) 

    #compare feature
    feature_compared = Average()([feature_lr, feature_hr])
    subs=Subtract()([feature_lr, feature_hr])


    # decoder
    conv6,n_filters=decoder_block(feature_compared, conv5, n_filters,upconv=upconv,droprate=droprate)
    print(n_filters)
    conv7,n_filters=decoder_block(conv6, conv4, n_filters,upconv=upconv,droprate=droprate)
    print(n_filters)
    conv8,n_filters=decoder_block(conv7, conv3, n_filters,upconv=upconv,droprate=droprate)
    print(n_filters)
    conv9,n_filters=decoder_block(conv8, conv2, n_filters,upconv=upconv,droprate=droprate)
    print(n_filters)    
    conv10,n_filters=decoder_block(conv9, conv1, n_filters,upconv=upconv,droprate=droprate)
    print(n_filters)

    conv11 = Conv2D(n_classes, (1, 1), activation='sigmoid')(conv10)

    model = Model(inputs=[input_lr,input_hr,input_near,input_far], outputs=[conv11,subs,mul])

    def weighted_binary_crossentropy(y_true, y_pred):
        class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
        return K.sum(class_loglosses * K.constant(class_weights))

    model.compile(optimizer=Adam(), loss=[weighted_binary_crossentropy,'MSE','MSE'],loss_weights=[1,0.8,0.1])
    model.summary()
    return model


if __name__ == '__main__':
	model = unet_model()
	# model.summary()
	with open('attentionmodel_summary.txt', 'w') as f:
		model.summary(print_fn=lambda x: f.write(x + '\n'))
	plot_model(model, to_file='attention.png', show_shapes=True)
