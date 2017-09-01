from setuptools import setup, find_packages

setup(
    name='pyace',
    version='0.1.1',
    description='A minimal python implementation of automatic chord estimation',
    author='Junqi Deng',
    author_email='dengjunqi06323011@gmail.com',
    url='https://github.com/tangkk/pyace',
    packages=find_packages(),
    long_description="""\
        This derives from my PhD thesis:
        Deng, J., Large Vocabulary Automatic Chord Estimation from Audio Using Deep Learning Approaches. PhD thesis, Department of Electrical and Electronic Engineering, The University of Hong Kong, 2016
        
        However the code of this work is written in matlab (https://github.com/tangkk/tangkk-mirex-ace). So I try to port those code into python. But this is by no means a direct porting. It is meant to be minimalist version of ACE, which keeps only the algorithmic gist of the original work.
        
        Compared with the original version which supports sevenths chords and inversions, this piece of code currently only supports maj and min triads, and it has much more elegant (only a few lines of) feature extraction and segmentation codes, which leverages librosa and hmmlearn.
    """,
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Topic :: Multimedia :: Sound/Audio",
    ],
    keywords='audio music chord mir',
    license='BSD',
    install_requires=[
        'numpy >= 1.7.0',
        'librosa',
        'hmmlearn',
        'theano',
        'keras'
    ],
)
