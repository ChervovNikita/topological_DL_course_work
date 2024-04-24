sudo apt update
sudo apt install python3-pip
pip3 install torch
pip3 install loguru
pip3 install pytorch_lightning
pip3 install lightning
pip3 install torchvision
pip3 install gudhi
pip3 install scipy
pip3 install scikit-learn
pip3 install matplotlib
pip3 install ripser
pip3 install diagram2vec
pip3 install jupyterlab
pip3 install -U 'tensorboardX'
pip3 install giotto-tda
pip3 install pandas 

cd experiments
python3 get_datasets_script.py
cd ..
python3 experiments/training.py data/mnist/images_train.pt data/mnist/images_test.pt trainable_conv_10 conv 16 300 '{"n_dims": [1, 10], "n_hidden": 64, "n_out": 10, "nhead": 2, "num_layers": 2, "dim_feedforward": 256, "device": "cpu"}'
