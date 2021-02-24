#imports
import argparse
import glob
import os

#3rd party imports
from keras import backend as K
from tensorflow.python.client import device_lib

#local imports
import file_arrange as fa
import DL_models



#check gpu is available

print(K.tensorflow_backend._get_available_gpus())
print(device_lib.list_local_devices())


# parse the commandline
parser = argparse.ArgumentParser(description='schizophrenia Deep learning binary classfication using keras.application')
parser.add_argument('--preprocessing', default=True, help='binary, For the first, you need this, but for the 2nd time it will not needed')

# data organization parameters
parser.add_argument('directory', type=str, help='base data directory')
parser.add_argument('--pp-standard', default=False, help='binary, whether divide dataset based on person number')
parser.add_argument('--cls-index', type=int, default=4, help='ex) fig_h03.edf_38.tif --> h index : 4' , metavar='classification digits')
parser.add_argument('--person-index', type=int, default=[5,6], nargs='+', help='ex) fig_h03.edf_38.tif --> 03 index : [5,6]' , metavar='person number digits')
# command line for this
# python train.py 'F:\\Dataset\\Schizophrenia_CNN\\image_tif' --person_digit 5 6 7

# training parameters
parser.add_argument('--train-ratio', '--tr',type=float, default=0.8, help='Training ratio')
parser.add_argument('--val-ratio','--val', type=float, default=0.1, help='validation ratio')
parser.add_argument('--test-ratio', type=float, default=0.1, help='test ration')
parser.add_argument('--test-val-comb', default=False, help='train = train_ratio, test&val = test_ratio+val_ratio' , metavar='combine test & valid dataset when arrange data')
parser.add_argument('--expln', default=True,  help='print explanation')
parser.add_argument('--weights', default='imagenet',  help='"imagenet or None" based on keras.application weight')
parser.add_argument('--batch-size', default=100,  help='batch size (default: 100)')
parser.add_argument('--epochs', default=30,  help='number of training epochs (default: 30)')
parser.add_argument('--gpu', default='0', help='GPU ID numbers (default: 0)')
parser.add_argument('--param_denselayer', default=[2048,512,2], nargs='+', help='dense layer parameter (default : [2048,512,1])')



args = parser.parse_args()



# dir_img = 'F:\\Dataset\\Schizophrenia_CNN\\image_tif'
# classification digit in the picture, count first from 0
# cls_digit = 4
# file_person_num_digit = [5,6]
# train_ratio =0.8
# val_ratio =0.1
# test_ratio =0.1
# test_val_combine = False
# expln=True

#whether you wanna meke preprocessing folder
preprocessing = False

# to check how long does it take to copy the dataset --> start point
fa.tic()
if args.preprocessing:
    dir_subdir_copy = fa.data_arrange(args.directory, pp_standard=args.pp_standard, file_cls_name_digit=args.cls_index,
                                      file_person_num_digit=args.person_index, train_ratio=args.train_ratio,
                                      val_ratio=args.val_ratio, test_ratio=args.test_ratio,
                                      test_val_combine=args.test_val_comb, expln=args.expln)
else:
    whole_dirs = fa.find_whole_subdir(args.directory)
    for dir_imgs in whole_dirs:
        dic_HS_num, dic_HS_list, dic_HS_patient = fa.about_dir(args.directory, file_cls_name_digit=args.cls_index,
                                                               file_person_num_digit=args.person_index, expln=args.expln)
    dir_subdir_copy = fa.mksubdirs(args.directory, list(dic_HS_num.keys()))

# to check how long does it take to copy the dataset --> end point
print('time for copying data')
fa.tac()



# # deep learning
# fa.tic()
# img_size= fa.size_check(args.directory)
# sizes = [size for size in img_size.keys()]
# input_shape = sizes[0]+(3,)
#
# DL_models.keras_whole_models(dir_subdir_copy, weights, batch_size=100, epochs=30, param_denselayer=[2048,512,1])
#
# print('time for copying data')
# fa.tac()









