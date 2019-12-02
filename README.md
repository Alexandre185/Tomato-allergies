# Tomato-allergies

## Requirements

### Files

The 3 files/directories needed to run the code are:
- the directory `assignment_imgs` with the 3000 images
- the file  `img_annotations.json` with all the bounding boxes for the images
- the file `Annotations_with_tomato.txt` with the ID of the bounding boxes that contain or may contain tomato.

### Packages

```python
# Packages 
import json
import os
import cv2
import numpy as np
import pandas as pd
```

## Annotations_with_tomato.txt file

The file Annotations_with_tomato.txt was made from the file label_mapping.csv. It consist in the list of the ID of the 76 elements that may contain tomatoes that I have identified in the label_mapping list.

## Code

First the path of the needed files need to be set and the `img_annotations.json` and `Annotations_with_tomato.txt` must be opened as follows:

```python
path_to_images = '/.../assignment_imgs/' # (unzipped)
path_to_annotations = '/.../img_annotations.json'
path_to_annotations_tomato = '/.../Annotations_with_tomato.txt'

# Reading of annotation and annotation_tomato files

with open(path_to_annotations) as json_file:
    annotations = json.load(json_file)

f = open(path_to_annotations_tomato, 'r')
annotations_tomato = f.read()
annotations_tomato = annotations_tomato.split('\n')
```

```python
# Creation of the labels dictionnary {image_ID : 0 or 1}
# 0 if the image does not have any tomato annotation
# 1 if the image contain at least 1 annoation with tomato

labels_dict =  {}
for i in range(len(annotations)): 
    labels_dict[list(annotations.items())[i][0]] = 0
    for bbox in list(annotations.items())[i][1]:
        if bbox['id'] in annotations_tomato:
            labels_dict[list(annotations.items())[i][0]] = 1
```

```python
# Creation of the images and labels arrays

images = []
labels = []

for ID in os.listdir(path_to_images):
    images.append(cv2.imread(path_to_images+ID))
    labels.append(labels_dict[ID])
```

```python
img_dim = 299 # Dimension we want to resize the image to (image is 600*600 initially)
# Reshape of the 13 images that are not 600*600 ( 10 are 600*601, 1 is 600*654, 1 is 600*664, 1 is 986*600)
for i in range(len(images)):
    if images[i].shape != (img_dim, img_dim, 3):
        images[i] = cv2.resize(images[i],(img_dim, img_dim))
```

```python
images = np.array(images, dtype="float") / 255.0  # not enough RAM with colab
labels = np.array(labels)
```
```python
# One hot encoding of the labels
labels = pd.get_dummies(labels).values
```
```
