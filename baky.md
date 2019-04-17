# Abstract

# Introduction
* motivation
    - A short text about autonomous cars, how they are the future and what they need to succeed
    - Where traffic sign recognition and classification fits in

### Traffic signs
* history
    - [Vienna Convention on Road Signs and Signals](https://en.wikipedia.org/wiki/Vienna_Convention_on_Road_Signs_and_Signals)

* what's special about them? (high contrast, basic shapes)
* in Czech Republic
    - categories (prohibitory, informative, ...)

### Traffic sign recognition (TSR)
* what it is?
* why make it two separate tasks
    - detection
    - classification
* challenges
    - light conditions
    - blur
    - dirt or graffiti
    - obstructed parts (this is somewhat similar to the previous item)
    - rotation
* related works
    - other approaches
        - color-based
        - shape-based
        - template matching
        - other machine learning  methods (SVM)
    - neural networks
        - CNNs
        - why they are good for image recognition?
        - [ImageNet](http://www.image-net.org/challenges/LSVRC/)
        - other publications about TSR (citations)
    - state of the art

### Data
* open data sets
    - short description of each
    - their features
        - german
        - belgian
        - italian
        - chinese
        - [telenav](https://github.com/Telenav/Telenav.AI)?
* table of features
* final data set
    - pictures of all
    - maybe histogram and other characteristics
    
### Goals of this thesis
* what's special about this thesis
* goals (very likely to change)
    - try to understand how CNNs learn traffic signs
    - which features are important
    - how to increase performance
    - how to measure performance
    - augmenting the data set
    
# Methodology

### Analyze the data set
* which signs I chose and why
* features of the data set

### My data set
* why?
* where?
* how?

### Processing the data set
* pre-processing
    - histogram equalization
    - gray scale
* augmentation
    - expand the data set
    - add missing conditions (blur, dirt, ...)
        - cropping
        - flipping
        - rotation
        - shearing
        - perspective transformation
        - color shift
        - blur
        - light effects (flares)
        - noise (why dropout is better in the model itself)

### Model
* language & framework (Python + TensorFlow/Keras)
* architecture
    - hyper parameters - which were fixed and why
* training
    - metacentrum
    
# Results and discussion

### Results
* TensorBoard graphs
* which was best
    - different metrics
    - what is a good metric?

### Discussion
* which signs were problematic (and why)
* visualization ([Lucid](https://github.com/tensorflow/lucid))

# Conclusion
* which goals were achieved
* what could be improved
    - future work (master thesis?)

# Attachments
- list of all traffic signs
- possibly other theses