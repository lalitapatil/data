# from google_images_download import google_images_download   #importing the library
#https://github.com/hardikvasa/google-images-download/blob/master/docs/examples.rst
response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":"Poultry & Seafood,Fruits,Vegetables","limit":100,"print_urls":True}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images
#https://towardsdatascience.com/https-medium-com-drchemlal-deep-learning-tutorial-1-f94156d79802
%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai.vision import *
from fastai.metrics import error_rate
bs = 5  #batch size: if your GPU is running out of memory, set a smaller batch size, i.e 16
sz = 224 #image size
PATH = 'C:/Users/omkar/Desktop/food'
classes = []
for d in os.listdir(PATH):
    if os.path.isdir(os.path.join(PATH, d)) and not d.startswith('.'):
        classes.append(d) 
print ("There are ", len(classes), "classes:\n", classes)    
for c in classes:
    print ("Class:", c)
    verify_images(os.path.join(PATH, c), delete=True)
data  = ImageDataBunch.from_folder(PATH, ds_tfms=get_transforms(), size=sz, bs=bs, valid_pct=0.2).normalize(imagenet_stats)
print ("There are", len(data.train_ds), "training images and", len(data.valid_ds), "validation images." )
data.show_batch(rows=3, figsize=(7,7))
learn = cnn_learner(data, models.resnet34, metrics=accuracy)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11), heatmap=False)
