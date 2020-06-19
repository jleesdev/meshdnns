# meshdnns
Four different mesh-specific dnn models (CoMA, SpiralNet++, MeshCNN, and MeshNet).

#### data samples
* format: .ply  
* details: v 2922, f 5840, e 8760.  
* template: the average of training samples.

#### training
* coma  
`$ python train_coma.py --train_data train_set.csv --test_data test_set.csv`  
 to configure the parameters, please consider **cfgs/coma.cfg** file. 

* spiralnet  
`$ python train_spiralnet.py --train_data train_set.csv --test_data test_set.csv`  
 to configure the parameters, please consider code lines (115-145) in **train_spiralnet.py** file. 

* meshnet  
`$ python utils/meshnet/preprocessing --train_data train_set.csv --test_data test_set.csv`  
`$ python train_meshnet.py --train_data train_set.csv --test_data test_set.csv`  
 to configure the parameters, please consider **cfgs/meshnet_train.yaml** file.  
 to change the labels (AD:0, CN:1), please consider the variable **type_to_index_map** in **datatsets/meshnet_dataset.py**.

* meshcnn
`$ python train_meshcnn.py --train_data train_set.csv --test_data test_set.csv`  
 to configure the parameters, please consider **utils/meshcnn/base_options.py** and **utils/meshcnn/train_options.py**  
 to change the labels (AD:0, CN:1), please consider the variable **type_to_index_map** in **datatsets/meshcnn_dataset.py**.
