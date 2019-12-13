Triplet loss for facial recognition.

# Triplet Face

The repository contains code for the application of triplet loss training to the
task of facial recognition. This code has been produced for a lecture and is not
going to be maintained in any sort.


## Travail effectué
Fork du projet de base.

###Sur colab :

Import du projet git
```
!git clone https://github.com/EmmanuelVey94/TripletFace.git
```
Ajout de mon drive sur colab
```
from google.colab import drive
drive.mount('/content/drive')
```
Extraction de l'archive du dataset
```
!unzip -F ../drive/My\ Drive/ESILV5/dataset2.zip
```
Installation des bibliothèques nécéssaire à l'entrainement du model
```
!pip3 install -r requirements.txt
```
Entrainement du model avec 4 époch. C'est suffisant pour obtenir un resultat utilisable pour la reconnaissance.
J'ai aussi tester le model avec resnet-152. 
```
!python3 -m tripletface.train -s dataset/ -m model -e 4 
```

Je vais maintenant utiliset un jit compile pour pouvoir utiliser rapidement le réseau de neurone sans avoir à le ré executer.

```
import torch
import torch.nn as nn
from tripletface.core.model import Encoder

model = Encoder(64)
weigths = torch.load( "/content/TripletFace/model/model.pt" )['model']
model.load_state_dict( weigths )
jit_model = torch.jit.trace( model,torch.rand(64, 3, 7, 7) )
torch.jit.save( jit_model, "jitmodel.pt" )
```
Il suffit maintenant de charger le jitmodel.pt pour pouvoir l'utiliser sur les images.

```
model = torch.jit.load("jitmodel.pt")
```
On va maintenant utiliser le réseau sur quelques images d'une personne pour déterminer les centroïdes et les thresholds




## Todo ( For the students )

**Deadline Decembre 13th 2019 at 12pm**

The students are asked to complete the following tasks:
* Fork the Project
* Improve the model by playing with Hyperparameters and by changing the Architecture ( may not use resnet )
* JIT compile the model ( see [Documentation](https://pytorch.org/docs/stable/jit.html#torch.jit.trace) )
* Add script to generate Centroids and Thesholds using few face images from one person
* Generate those for each of the student included in the dataset
* Add inference script in order to use the final model
* Change README.md in order to include the student choices explained and a table containing the Centroids and Thesholds for each student of the dataset with a vizualisation ( See the one above )
* Send the github link by mail
