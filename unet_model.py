# u-net model with up-convolution or up-sampling and weighted binary-crossentropy as loss func

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout,Average,Subtract,Multiply,Lambda
from keras.optimizers import Adam,SGD,RMSprop
from keras.utils import plot_model
from keras import backend as K           


def conv_block(input, num_filters):
    conv = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(input)
    conv = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(conv)
    return conv

def encoder_block(input, num_filters,droprate=0.25,dropout=True,growth_factor=2):
    conv = conv_block(input, num_filters)
    pool = MaxPooling2D((2, 2))(conv)
    if dropout:
        pool = Dropout(droprate)(pool)
    pool = BatchNormalization()(pool)
    num_filters*=growth_factor
    return conv, pool,num_filters

def decoder_block(input, concat_tensor, num_filters,upconv=True,droprate=0.25,growth_factor=2):
    num_filters//= growth_factor
    if upconv:
        x = concatenate([Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input), concat_tensor])
    else:
        x = concatenate([UpSampling2D((2, 2))(input), concat_tensor])
    x=BatchNormalization()(x)
    x = conv_block(x, num_filters)
    x=  Dropout(droprate)(x)
    return x,num_filters

def  encoder(input, n_filters_start=32, droprate=0.25):
    layer_outputs = []
    x=input
    n_filters=n_filters_start  
    for i in range(5):  
        conv, pool, n_filters = encoder_block(x, n_filters, droprate=droprate)
        layer_outputs.append(conv)  
        x = pool
        print(n_filters)
        if i == 4:
            feature=conv_block(pool,n_filters)
    return layer_outputs,feature

def decoder(input, layer_outputs, n_filters_start=32, droprate=0.25,n_classes=4,upconv=True):
    x=input
    n_filters=n_filters_start  
    for i in range(5):
        x,n_filters=decoder_block(x, layer_outputs[4-i], n_filters,upconv=upconv,droprate=droprate)
        print(n_filters)
        if i == 4:
            output=Conv2D(n_classes, (1, 1), activation='sigmoid')(x)
    return output

def unet_model(n_classes=4, im_sz=32, n_channels=3, n_filters_start=32, growth_factor=2, upconv=True,
               class_weights=[0.2, 0.2, 0.5, 0.1]):
    droprate=0.25
    input_lr = Input((im_sz, im_sz, n_channels))
    input_hr = Input((im_sz, im_sz, n_channels))
    input_near = Input((im_sz, im_sz, n_channels))
    input_far = Input((im_sz, im_sz, n_channels))
    
    # encoder for all lr
    lr_convs,feature_lr = encoder(input_lr, n_filters_start, droprate=droprate)
    # encoder for hr
    hr_convs,feature_hr = encoder(input_hr, n_filters_start, droprate=droprate)
   

    #submodel
    submodel = Model(inputs=input_lr, outputs=[feature_lr] + lr_convs)
    outputs_lr_n = submodel(input_near)  
    outputs_lr_f = submodel(input_far) 
    feature_lr_n = outputs_lr_n[0]
    feature_lr_f = outputs_lr_f[0]
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
    output=decoder(feature_compared, lr_convs, n_filters_start, droprate=droprate,n_classes=n_classes,upconv=upconv)

    model = Model(inputs=[input_lr,input_hr,input_near,input_far], outputs=[output,subs,mul])

    def weighted_binary_crossentropy(y_true, y_pred):
        class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
        return K.sum(class_loglosses * K.constant(class_weights))

    model.compile(optimizer=Adam(), loss=[weighted_binary_crossentropy,'MSE','MSE'],loss_weights=[1,0.8,0.1])
    model.summary()
    return model


if __name__ == '__main__':
    model = unet_model()
    # model.summary()
    with open('model2_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    plot_model(model, to_file='model2.png', show_shapes=True)
