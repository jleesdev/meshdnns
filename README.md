# meshdnns
Four different mesh-specific dnn models (CoMA, SpiralNet++, MeshCNN, and MeshNet).

#### data samples
* format: .ply  
* details: v 2922, f 5840, e 8760.  
* template: the average of training samples.

#### training
* coma  
`$ python train_coma.py --train_data train_set.csv --test_data test_set.csv`  
* spiralnet  
`$ python train_spiralnet.py --train_data train_set.csv --test_data test_set.csv`  
