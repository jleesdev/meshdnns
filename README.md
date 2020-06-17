# meshdnns
Four different mesh-specific dnn models (CoMA, SpiralNet++, MeshCNN, and MeshNet).

### data samples
.ply format.
details: V 2922, F 5840, E 8760.

#### data preprocessing
* coma  
`$ python datasets/coma_dataset.py -dt train -d trainset.csv`  
`$ python datasets/coma_dataset.py -dt test -d testset.csv`
* spiralnet
