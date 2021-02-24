import setuptools

setuptools.setup(
    name='schizo_CNN',
    version='0.1',
    license='gpl-3.0',
    description='Schizophrenia image data Classification using keras.applications',
    url='https://github.com/yoonguusong/Schizo_CNN',
    keywords=['Schizophrenia','classification', 'imaging', 'tif', 'keras'],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'tensorflow',
        'keras',
        'scikit-image',
        'matplotlib',
        'numpy',
        'scipy',
        'shutil',
    ]
)