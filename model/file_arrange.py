'''
This is for Kyungwon Kim
'''
import os
import numpy as np
import random
from shutil import copyfile, move
import copy
import time
import tensorflow as tf


def data_arrange(dir, pp_standard=False, file_cls_name_digit = 4, file_person_num_digit=[5,6], train_ratio = 0.8, val_ratio = 0.1, test_ratio=0.1, test_val_combine=False,  expln = True ):
    '''
    to arrange dataset into training, test, valid set

    :param dir: directory of your image dataset
    :param file_cls_name_digit: classification digits
                      ex) fig_h03.edf_38.tif --> h location --> 4 cuz it counts from 0
    :param file_person_num_digit: person number digits
                      ex) fig_h03.edf_38.tif --> 03 location --> [5,6] cuz it counts from 0
    :param expln: To print the explanation
    :param pp_standard: class standard whether divide dataset based on person or not
                      ex) pp_standard is True --> in 10 person, divide 9 training set, 1 test set
                      ex) pp_standard is False --> divide dataset in the ration of 9 training set, 1 test set
    :param Train_val_ratio: Training ratio
    :param val_ratio: validation ratio
    :param test_ratio: test ratio
    :param test_val_combine: binary, decide whether test & validation set will be combined when divide dataset into training, test, validation
    :param explain: to print explanation on dataset

    :return: subdirectory of arranged dataset
    '''

    # dir = 'F:\\Dataset\\Schizophrenia_CNN\\image_tif'
    # file_cls_name_digit = 4
    # file_person_num_digit = [5, 6]
    # expln = True
    # pp_standard = False
    # train_ratio = 0.8
    # val_ratio = 0.1
    # test_ratio = 0.1
    # test_val_combine = True
    # test_val_combine = False


    whole_dirs = find_whole_subdir(dir)
    for dir_imgs in whole_dirs:
        dic_HS_num, dic_HS_list, dic_HS_patient=about_dir(dir_imgs, file_cls_name_digit=file_cls_name_digit, file_person_num_digit=file_person_num_digit, expln=expln)

        # To divide dataset based on pp_standard
        # Train_ratio = 8, val_ratio = 1, test_ratio = 1
        if pp_standard:
            # make subdirectory "class\Train" & "class\Validation" & "class\Test" & copy the each dataset
            # 사람기준 numbering
            dir_subdir_copy = mksubdirs(dir_imgs, list(dic_HS_num.keys()))
            copy_files_PB(dir_imgs, dic_HS_list, dir_subdir_copy, dic_HS_patient,train_pct=train_ratio, test_pct=val_ratio,
                          val_pct=test_ratio, test_val_combine=test_val_combine, explain=expln)

            if expln:
                print('dataset classification is not based on people number')

        # Do not divide the dataset based on people numbering standard
        else:
            #make subdirectory "class\Train" & "class\Validation" & "class\Test" & copy the each dataset
            dir_subdir_copy = mksubdirs(dir_imgs, list(dic_HS_num.keys()))
            copy_files_notPB(dir_imgs, dic_HS_list, dir_subdir_copy, train_pct=train_ratio, test_pct=val_ratio,
                       val_pct=test_ratio, test_val_combine=test_val_combine, explain=expln)
            if expln:
                print('dataset classification is not based on people number')

    return dir_subdir_copy


def about_dir(dirs, file_cls_name_digit = 4, file_person_num_digit=[5,6], expln = False):
    '''
     return dictionary values including dir file's classification file data

    :param dirs: directory
    :param file_cls_name_digit: classification digits
                           ex) fig_h03.edf_38.tif --> h location --> 4 cuz it counts from 0
    :param file_person_num_digit: person number digits
                           ex) fig_h03.edf_38.tif --> 03 location --> [5,6] cuz it counts from 0
    :param expln: explanation
    :return:
        dic_HS_num :
            {'h': 4694, 's': 4782}
        dic_HS_list :
            {'h': ['fig_h03.edf_281.tif', 'fig_h13.edf_215.tif', 's': 'fig_s08.edf_272.tif', 'fig_s14.edf_3.tif'}
        dic_HS_patient:
            {'h': {'01', '02', '08', '05', '10', '03', '04', '06', '09', '11', '12', '13', '14', '07'},
            's': {'01', '02', '08', '05', '10', '03', '04', '06', '09', '11', '12', '13', '14', '07'}}
    '''
    sch_DTs = os.listdir(dirs)

    # To count the dataset in classification, H: Healthy, S:Schizophrenia
    dic_HS_num = {}

    # dictionary for healthy and schizo to save the data list
    dic_HS_list = {}

    # dictionary for Healthy & Schizo patient number
    dic_HS_patient = {}

    for sch_dt in sch_DTs:
        cls_digit = sch_dt[file_cls_name_digit]  # s or #h
        num_patient = sch_dt[file_person_num_digit[0]:file_person_num_digit[0] + len(file_person_num_digit)]
        # print('num_patient : ',num_patient, 'sch_dt', sch_dt )

        if cls_digit in dic_HS_num:
            dic_HS_num[cls_digit] = dic_HS_num[cls_digit] + 1
            dic_HS_list[cls_digit].append(sch_dt)
            dic_HS_patient[cls_digit].add(num_patient)
        else:
            dic_HS_num[cls_digit] = 1
            dic_HS_list[cls_digit] = [sch_dt]
            dic_HS_patient[cls_digit] = set()
            dic_HS_patient[cls_digit].add(num_patient)

    if expln:
        print('num of Schizophrenia dataset : ', len(sch_DTs))
        print('In your dataset, classification name are : ', dic_HS_num)

    return dic_HS_num, dic_HS_list, dic_HS_patient


def copy_files_PB(dir, src, dst, nums_pp, file_person_num_digit=[5,6], train_pct=0.8, test_pct=0.1,
                                                  val_pct=0.1, test_val_combine=False, explain=False):
    '''
    copy files in 'dir' variable directory in the ration of train, test, validation percentage
    not based on Patient Number

    just in the copy_files_PB fuction,
    input variable nums_pp is added

    :param dir: directory
    :param nums_pp: patient's number dictionary
    ex) {'h': {'01', '02', '08', '05', '10', '03', '04', '06', '09', '11', '12', '13', '14', '07'},
         's': {'01', '02', '08', '05', '10', '03', '04', '06', '09', '11', '12', '13', '14', '07'}}

    :param dst: list of destination directory
    :param train_pct: train data percentage
    :param test_pct: test data percentage
    :param val_pct: validation data percentage
    :param test_val_combine: decide whether test data and validation data combine
    :param explain : to print explanation on dataset
    :return:
    '''




    # dir = 'F:\\Dataset\\Schizophrenia_CNN\\image_tif'
    # src= dic_HS_list
    # dst= dir_subdir_copy
    # nums_pp= dic_HS_patient
    # train_pct = 0.8
    # test_pct = 0.1
    # val_pct = 0.1
    # test_val_combine = False
    # explain = True
    # explain = False

    for cls in list(nums_pp.keys()):
        sub_cls_folders = [sub_cls for sub_cls in dst if os.path.basename(sub_cls) == cls]
        dir_dst_tr = [x for x in sub_cls_folders if 'Train' in x][0]
        dir_dst_val = [x for x in sub_cls_folders if 'Validation' in x][0]
        dir_dst_tst = [x for x in sub_cls_folders if 'Test' in x][0]

        #shuffle list to randomly select training, val, test data


        num_tr, num_val, num_tst = data_divider(len(nums_pp[cls]), train_pct=train_pct, test_pct=test_pct,
                                              val_pct=val_pct, test_val_combine=test_val_combine)

        num_trains = set(random.sample(list(nums_pp[cls]), num_tr))
        left = nums_pp[cls] - num_trains
        num_vals = set(random.sample(list(left), num_val))
        num_tsts = left - num_vals




        if explain:
            print('-------------before copy the data-------------')
            print('cls : ', cls)
            print('dir_dst_tr :', dir_dst_tr, ' dir_dst_val :', dir_dst_val, ' dir_dst_tst : ', dir_dst_tst)
            print('the number of training   patients numbers : ', num_trains)
            print('the number of validation patients numbers : ', num_vals)
            print('the number of   test     patients numbers : ', num_tsts)
        # to copy files, decide directory of copy source, copy destination
        for file in src[cls]:
            dir_src = os.path.join(dir, file)

            #decide destination directory in terms of file number index in list
            #file's patient's number
            num_patient = file[file_person_num_digit[0]:file_person_num_digit[0]+len(file_person_num_digit)]
            if num_patient in num_trains:
                dir_dst = os.path.join(dir_dst_tr, file)
            elif num_patient in num_vals:
                dir_dst = os.path.join(dir_dst_val, file)
            else:
                dir_dst = os.path.join(dir_dst_tst, file)

            # move(dir_src, dir_dst)
            copyfile_nonerror(dir_src, dir_dst)

    if explain:
        for subdir in dst:
            print('-------------after copy the data-------------')
            print(subdir, '\t file number : ', len(os.listdir(subdir)))




def copy_files_notPB(dir, src, dst, train_pct=0.8, test_pct=0.1,
                                                  val_pct=0.1, test_val_combine=False, explain=False):
    '''
    copy files in 'dir' variable directory in the ration of train, test, validation percentage
    not based on Patient Number

    :param dir: directory
    :param src: dictionary data ex: {'h': ['fig_h02.edf_121.tif', 'fig_h05.edf_135.tif', ..]}
    :param dst: list of destination directory
    :param train_pct: train data percentage
    :param test_pct: test data percentage
    :param val_pct: validation data percentage
    :param test_val_combine: decide whether test data and validation data combine
    :param explain : to print explanation on dataset
    :return:
    '''




    # dir = 'F:\\Dataset\\Schizophrenia_CNN\\image_tif'
    # src= dic_HS_list
    # dst= dir_subdir_copy
    # train_pct = 0.8
    # test_pct = 0.1
    # val_pct = 0.1
    # test_val_combine = False
    # explain = True

    for cls in list(src.keys()):
        sub_cls_folders = [sub_cls for sub_cls in dst if os.path.basename(sub_cls) == cls]
        dir_dst_tr = [x for x in sub_cls_folders if 'Train' in x][0]
        dir_dst_val = [x for x in sub_cls_folders if 'Validation' in x][0]
        dir_dst_tst = [x for x in sub_cls_folders if 'Test' in x][0]

        #shuffle list to randomly select training, val, test data
        random.shuffle(src[cls])
        num_tr, num_val, num_tst = data_divider(len(src[cls]), train_pct=train_pct, test_pct=test_pct,
                                              val_pct=val_pct, test_val_combine=test_val_combine)
        if explain:
            print('-------------before copy the data-------------')
            print('cls : ', cls)
            print('dir_dst_tr :', dir_dst_tr, ' dir_dst_val :', dir_dst_val, ' dir_dst_tst : ', dir_dst_tst)
            print('num_tr, num_val, num_tst : ', num_tr, num_val, num_tst)

        # to copy files, decide directory of copy source, copy destination
        for file in src[cls]:
            dir_src = os.path.join(dir, file)

            #decide destination directory in terms of file number index in list
            if src[cls].index(file) < num_tr:
                dir_dst = os.path.join(dir_dst_tr, file)
            elif src[cls].index(file) < num_tr+num_val:
                dir_dst = os.path.join(dir_dst_val, file)
            else:
                dir_dst = os.path.join(dir_dst_tst, file)

            # move(dir_src, dir_dst)
            copyfile_nonerror(dir_src, dir_dst)

    if explain:
        for subdir in dst:
            print('-------------after copy the data-------------')
            print(subdir, '\t file number : ', len(os.listdir(subdir)))




def mksubdirs(dir, sub_category, comment=False, test_val_combine = False):
    '''
    if directory A is given,
    "A_copy" directory is made & subdirectory of classification "A_copy/cls1", "A_copy/cls2", "A_copy/cls3" is made

    :param dir: directory
    :param cls_s: list value, classifications
    :return: list of all new made absolute path of subdirectories
    '''

    if test_val_combine :
        ls = ['Train', 'Test']
    else:
        ls = ['Train', 'Validation', 'Test']

    dir_copy_fold = os.path.join(os.path.dirname(dir), os.path.basename(dir) + "_copy")
    mkdir_nonerror(dir_copy_fold)

    dir_subs =[]
    for tvt in ls:
        dir_tvt = os.path.join(dir_copy_fold, tvt)
        mkdir_nonerror(dir_tvt)
        for sub in sub_category:
            dir_sub = os.path.join(dir_tvt, sub)
            mkdir_nonerror(dir_sub)
            dir_subs.append(dir_sub)
    return dir_subs

def copyfile_nonerror(src, dst, print_comment = False):
    '''
    make directory
    if directory is exist, just pass
    if directory is not exist, this will make directory
    :param dir: directory to be make
    :param print_comment: binary for commenting
    :return:
    '''
    # src='F:\\Dataset\\Schizophrenia_CNN\\image_tif\\fig_s04.edf_154.tif'
    # dst= 'F:\\Dataset\\Schizophrenia_CNN\\image_tif_copy\\Test\\s\\fig_s01.edf_6.tif'

    if os.path.exists(dst):
        if print_comment == True:
            print('already exist : %s' % dst)
    else:
        copyfile(src, dst)
        if print_comment == True:
            print('src ----> dst : %s ' % dst)
        # pass


def mkdir_nonerror(dir, comment = False):
    '''
    make directory
    if directory is exist, just pass
    if directory is not exist, this will make directory
    :param dir: directory to be make
    :param comment: binary for commenting
    :return:
    '''

    if not os.path.isdir(dir):
        os.mkdir(dir)
        if comment == True:
            print('     new      : %s '%dir)
    else:
        if comment == True:
            print('already exist : %s'%dir)
        pass



def read_folder(dir_root):
    '''
    This is for recursive function
    :param dir:directory for images
    :return:
    '''

    dirs_branch, nondirs_branch = [],[]
    list_folder = os.listdir(dir_root)
    for fold in list_folder:
        if os.path.isdir(os.path.join(dir_root, fold)):
            dirs_branch.append(fold)
        else:
            nondirs_branch.append(fold)

    yield dir_root, dirs_branch, nondirs_branch

    for fold in dirs_branch:
        new_path = os.path.join(dir_root,fold)
        for x in read_folder(new_path):
            yield x

def find_whole_subdir(dir):
    '''
    This is for finding directory which has dcm files
    :param dir: directory
    :return: list of directories which has dcm files
    '''

    dir_dcm = []
    for dir_root, dir_bc, dir_nonbc in read_folder(dir):
        subdir_exist_flag = False
        for fold in os.listdir(dir_root):
            new_path = os.path.join(dir_root, fold)
            if os.path.isdir(new_path):
                subdir_exist_flag = True
        if subdir_exist_flag == False:
            dir_dcm.append(dir_root)
    return dir_dcm

def data_divider(data_size, train_pct=0.8, test_pct=0.1, val_pct=0.1, test_val_combine=True):
    '''
    divide the dataset into training, validation, test set

    :param data_size: data lists
    :param train_pct: train data percentage
    :param test_pct: test data percentage
    :param val_pct: validation data percentage
    :param test_val_combine: decide whether test data and validation data combine
    :return: divided data_train, data_val, data_test will be returned
    '''
    # data_size = 4782
    # train_pct = 0.8
    # test_pct = 0.1
    # val_pct = 0.1
    assert train_pct+test_pct+val_pct==1, 'sum of training, validation, test percentage is not 1'

    num_train = round(data_size *train_pct) #7581
    num_val = round(data_size *val_pct) #948
    num_test = data_size - (num_train +num_val) #947

    assert data_size == (num_train +num_test +num_val), 'training, validation, test number is not matched'
    if test_val_combine ==True:
        num_val = num_val+num_test
        num_test =0

    return num_train, num_val, num_test



def tic():
    '''
    for checking processing time
    this is for beginning
    :return:
    '''
    global _start_time
    _start_time = time.time()

def tac():
    '''
    for checking processing time
    this is for ending
    :return:
    '''
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec))


def size_check(dir, expln=True):
    '''
    for checking image sizes in the specific directory
    :param dir: directory of image
    :param expln: binary, explain
    :return:
    '''
    from PIL import Image
    # check file image size is constant
    img_size = {}
    for img in os.listdir(dir):
        path = os.path.join(dir, img)
        im = Image.open(path)
        if im.size in img_size:
            img_size[im.size] += 1
        else:
            img_size[im.size] = 0
    if expln:
        print('image file size : ',img_size)
    assert img_size.keys() != 1, 'file size is not matched,file sizes : {} file_numbers : {} each'.format(
        [key for key in img_size.keys()], [value for value in img_size.values()])

    return img_size








