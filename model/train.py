#import 3rd paety


#local imports
from model import DL_models, file_arrange as fa

#To use the code, change the following variables in your system
#directory of dataset
dir_img = 'F:\\Dataset\\Schizophrenia_CNN\\image_tif'

#classification digit in the picture, count first from 0
cls_digit = 4
file_person_num_digit = [5,6]
train_ratio =0.8
val_ratio =0.1
test_ratio =0.1
test_val_combine = False
expln=True

#whether you wanna meke preprocessing folder
preprocessing = False

# to check how long does it take to copy the dataset --> start point
fa.tic()
if preprocessing:
    dir_subdir_copy = fa.data_arrange(dir_img, pp_standard=False, file_cls_name_digit=cls_digit,
                                      file_person_num_digit=file_person_num_digit, train_ratio=train_ratio,
                                      val_ratio=val_ratio, test_ratio=test_ratio,
                                      test_val_combine=test_val_combine, expln=expln)
else:
    whole_dirs = fa.find_whole_subdir(dir_img)
    for dir_imgs in whole_dirs:
        dic_HS_num, dic_HS_list, dic_HS_patient = fa.about_dir(dir_imgs, file_cls_name_digit=cls_digit,
                                                               file_person_num_digit=file_person_num_digit, expln=expln)
    dir_subdir_copy = fa.mksubdirs(dir_imgs, list(dic_HS_num.keys()))
# to check how long does it take to copy the dataset --> end point
print('time for copying data')
fa.tac()



# deep learning
fa.tic()
img_size= fa.size_check(dir_img)
sizes = [size for size in img_size.keys()]
input_shape = sizes[0]+(3,)

DL_models.DL_models(dir_subdir_copy)

print('time for copying data')
fa.tac()








