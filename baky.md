# Abstract
# Intro
- traffic sign recognition for autonomous cars
- why make it two separate tasks? (detection, classification)
- what's special about this thesis? (data set insight)

# Related works
## Other approaches
- color-based
- shape-based
- template matching
- other machine learning  methods (SVM)
## Neural networks (CNNs)
- why they are good for image recognition?
- [ImageNet](http://www.image-net.org/challenges/LSVRC/)
- other publications about TSR (citations)
- state of the art

# Data set
## Traffic signs
- what's special about them? (high contrast, basic shapes)
- categories (prohibitory, informative, ...)
- [Vienna Convention on Road Signs and Signals](https://en.wikipedia.org/wiki/Vienna_Convention_on_Road_Signs_and_Signals)[link](http://www.unece.org/fileadmin/DAM/trans/conventn/signalse.pdf)
- emphasize that the main use application is for autonomous vehicles
- possibly what's special about Czech signs
## Used data sets - which signs were selected
- german
- belgian
- italian
- chinese
- [telenav](https://github.com/Telenav/Telenav.AI)?
## My (czech) data set
- why?
- where?
- how?
## Final data set
**TODO: There needs to be way more here**
- which traffic signs I chose and why
### pre-processing
- histogram equalization
- gray scale
### augmentation
- cropping
- flipping
- rotation
- shearing
- perspective transformation
- color shift
- blur
- light effects (flares)
- noise (why dropout is better in the model itself)

# Model
## Methodology
- language & framework (Python + TensorFlow/Keras)
- hyper parameters - which were fixed and why
## Results
- training graphs (TensorBoard)
- which was best (by what metric?)

# Discussion and Summary
- which signs were problematic (and why)
- visualization ([Lucid](https://github.com/tensorflow/lucid))
- what could be improved, future work (master thesis?)

# Attachments
- list of all traffic signs
- possibly other theses
