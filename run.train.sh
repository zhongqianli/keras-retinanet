python keras_retinanet/bin/train.py --gpu 0 --snapshot-path ./snapshots --backbone resnet50 --batch-size 1 --epochs 100 --steps 1 --tensorboard-dir=./log --freeze-backbone --weights ~/.keras/models/ResNet-50-model.keras.h5 csv images.train.csv classes.csv