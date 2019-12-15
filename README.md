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
J'ai joué avec certains des hyper paramètres comme le nombre d'époch ou bien la taille du batch ainsi que le learning rate. Cependant il est difficile d'observer avec quels paramètres le model est la plus performant puisque je n'ai pas réussi a utiliser le model.

```
!python3 -m tripletface.train -s dataset/ -m model -e 4 
```
Une fois exécuté on obtien un fichier model.pt qui contient tout les poids du model. Ainsi que l'ensemble des rendus graphique de la triplet Loss. Pour 4 epoch j'ai donc obtenu 4 rendus.



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

Voici un essaie infructueux des nombreuses chose que j'ai essayé de faire pour créer les centroïdes et les thresholds sur une image :
```
import torch
import torch.nn as nn
import os
from tripletface.core.dataset import ImageFolder
from torchvision import transforms

model = torch.jit.load("jitmodel.pt")

trans         = {
    'train': transforms.Compose( [
        transforms.RandomRotation( degrees = 360 ),
        transforms.Resize( size = 224 ),
        transforms.RandomCrop( size = 224 ),
        transforms.RandomVerticalFlip( p = 0.5 ),
        transforms.ColorJitter( brightness = .2, contrast = .2, saturation = .2, hue = .1 ),
        transforms.ToTensor( ),
        transforms.Lambda( lambda X: X * ( 1. - noise ) + torch.randn( X.shape ) * noise ),
        transforms.Normalize( [ 0.485, 0.456, 0.406 ], [ 0.229, 0.224, 0.225 ] )
    ] ),
    'test':transforms.Compose( [
        transforms.Resize( size = 224 ),
        transforms.CenterCrop( size = 224 ),
        transforms.ToTensor( ),
        transforms.Normalize( [ 0.485, 0.456, 0.406 ], [ 0.229, 0.224, 0.225 ] )
    ] )
}
images = ImageFolder( os.path.join( "dataset/", 'test'  ), trans[ 'test'  ] )
model.eval(images[0][0])
```
Mais sans succès. Je n'arrive pas a "utiliser" le model sur l'image. Même après de longues recherches sur le net.



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
