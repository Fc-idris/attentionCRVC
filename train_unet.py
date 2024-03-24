from datetime import datetime
from unet_model import *
from gen_patches import *

import os
import os.path
import numpy as np
import tifffile as tiff
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

current_time = datetime.now().strftime("%m%d_%H%M")
os.environ['CUDA_VISIBLE_DEVICES']='0'

def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x


N_BANDS = 3 
N_CLASSES = 4 
CLASS_WEIGHTS = [0.3, 0.2, 0.49, 0.01]  
N_EPOCHS = 400
UPCONV = True
PATCH_SZ = 32   # should divide by 16
BATCH_SIZE = 32 
TRAIN_SZ = 4000 
VAL_SZ = 1000   



def get_model():
    return unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS, upconv=UPCONV, class_weights=CLASS_WEIGHTS)


weights_path = 'weights'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
weights_path += '/unet_weights.hdf5'

trainIds = [str(i).zfill(2) for i in range(3, 15)]  # all availiable ids: from "01" to "24"


if __name__ == '__main__':
    data_dirs = ['./data/mband', './data/mband2', './data/near', './data/far']
    gt_dirs = ['./data/gt_mband', './data/gt_mband2', './data/gt_mband2', './data/gt_mband2']
    column_names = ['LR', 'HR', 'LR_near', 'LR_far']

    X_train = {name: [] for name in column_names}
    Y_train = {name: [] for name in column_names}
    X_val = {name: [] for name in column_names}
    Y_val = {name: [] for name in column_names}

    for i, name in enumerate(column_names):
        X_train[name] = {}
        X_val[name] = {}
        Y_train[name] = {}
        Y_val[name] = {}
        for img_id in trainIds:
            img_path = '{}/{}.tif'.format(data_dirs[i], img_id)
            mask_path = '{}/{}.tif'.format(gt_dirs[i], img_id)

            img_m = normalize(tiff.imread(img_path).transpose([0,1,2]))
            mask = tiff.imread(mask_path).transpose([0,1,2]) / 255
            train_sz = int(3/4 * img_m.shape[0])  # use 75% of image as train and 25% for validation'
            print(img_m.shape)
            X_tr=img_m[:train_sz, :, :]
            X_v=img_m[train_sz:, :, :]
            Y_tr=mask[:train_sz, :, :]
            Y_v=mask[train_sz:, :, :]
            X_train[name][img_id] = X_tr
            X_val[name][img_id] = X_v
            Y_train[name][img_id] = Y_tr
            Y_val[name][img_id] = Y_v
        print(f'Images {name} were read')
    


    def train_net():
        print("start train net")
        x_lr_train, y_lr_train = get_patches(X_train['LR'], Y_train['LR'], n_patches=TRAIN_SZ, sz=PATCH_SZ)
        x_lr_val, y_lr_val = get_patches(X_val['LR'], Y_val['LR'], n_patches=VAL_SZ, sz=PATCH_SZ)
        x_hr_train, y_hr_train = get_patches(X_train['HR'], Y_train['HR'], n_patches=TRAIN_SZ, sz=PATCH_SZ)
        x_hr_val, y_hr_val = get_patches(X_val['HR'], Y_val['HR'], n_patches=VAL_SZ, sz=PATCH_SZ)
        x_near_train, y_near_train = get_patches(X_train['LR_near'], Y_train['LR_near'], n_patches=TRAIN_SZ, sz=PATCH_SZ)
        x_near_val, y_near_val = get_patches(X_val['LR_near'], Y_val['LR_near'], n_patches=VAL_SZ, sz=PATCH_SZ)
        x_far_train, y_far_train = get_patches(X_train['LR_far'], Y_train['LR_far'], n_patches=TRAIN_SZ, sz=PATCH_SZ)
        x_far_val, y_far_val = get_patches(X_val['LR_far'], Y_val['LR_far'], n_patches=VAL_SZ, sz=PATCH_SZ)

        
        y_sub=np.zeros((x_lr_train.shape[0],1,1,1024))
        y_sub2=np.zeros((x_lr_val.shape[0],1,1,1024))
        y_mul=np.ones((x_lr_train.shape[0],1,1,1024))
        y_mul2=np.ones((x_lr_val.shape[0],1,1,1024))
        
        model = get_model()
        if os.path.isfile(weights_path):
            model.load_weights(weights_path)
        #model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_weights_only=True, save_best_only=True)
        #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
        #reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0.00001)
        model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
        csv_logger = CSVLogger(f'log_unet_{current_time}.csv', append=True, separator=';')
        tensorboard = TensorBoard(log_dir=os.path.abspath('./tensorboard_unet/'), write_graph=True, write_images=True)

        # tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)
        model.fit([x_lr_train,x_hr_train,x_near_train,x_far_train], [y_lr_train,y_sub,y_sub], batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                  verbose=2, shuffle=True,
                  callbacks=[model_checkpoint, csv_logger, tensorboard],
                  validation_data=([x_lr_val,x_hr_val,x_near_val,x_far_val], [y_lr_val,y_sub2,y_sub2]))
        return model

    # train_net()
