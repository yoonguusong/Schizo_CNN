# Schizophrenia classification on image dataset  

**schizoCNN** is a general purpose library for learning-based tools for classification on existing Deep learning architecture on keras application.

# Tutorial

contact to [Schizo_CNN developer](https://github.com/yoonguusong) to learn about this algorithm


# Instructions

To use the Schizo_CNN library, either clone this repository and install the requirements listed in `setup.py` ~~or install directly with pip.~~

## Preprocessing
Dataset is shown in below.  
1st : healthy subject, 2nd : schizophrenia subject  
<img src="./data/data_example_healthy.jpg" width="40%" alt="healthy subject image"> <img src="./data/data_example_schizophrenia.jpg" width="40%" alt="schizophrenia subject image">

To understand how to divide the dataset, please look inside the code `./model/file_arrange.py`. This code arrange the dataset which contains in one directory by dividing into class in terms of data file name. You can divide the dataset in terms of subjects number or not caring the subjects number.
```
dataset(original)
    
```

```
dataset(preprocessed)
    - training
        - class s
        - class h
    - validation
        - class s
        - class h
    - test
        - class s
        - class h
```

This training, validation, test ratio can be adjusted.

## Training
on keras.application, there are ['DenseNet121', 'DenseNet169', 'DenseNet201', 'InceptionResNetV2', 'InceptionV3', 
'MobileNet', 'MobileNetV2', 'NASNetLarge', 'NASNetMobile', 'ResNet101', 'ResNet101V2', 'ResNet152', 'ResNet152V2',
 'ResNet50', 'ResNet50V2', 'VGG16', 'VGG19', 'Xception', 'keras_modules_injection']


## Testing 



# Contact:
For any problems or questions please contact email <syg7949@naver.com>  
