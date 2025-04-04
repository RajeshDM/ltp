# Running Multi Head attention with 1 heads

#Dataset Size : Small

#Attention dropout : 0
#python main.py --all-problems --expid blocks --domain ManyBlocks_ipcc_big --num-train-problems 200 --epochs 1000 --mode train --data-augmentation False --n-heads 1 --attention-dropout 0
#Attention dropout : 0.5
#python main.py --all-problems --expid blocks --domain ManyBlocks_ipcc_big --num-train-problems 200 --epochs 1000 --mode train --data-augmentation False --n-heads 1 --attention-dropout 0.5

#Dataset Size : Large

#Attention dropout : 0
python main.py --all-problems --expid blocks --domain ManyBlocks_ipcc_big --num-train-problems 200 --epochs 1000 --mode train --data-augmentation True --n-heads 1 --attention-dropout 0
#Attention dropout : 0.5
#python main.py --all-problems --expid blocks --domain ManyBlocks_ipcc_big --num-train-problems 200 --epochs 1000 --mode train --data-augmentation True --n-heads 1 --attention-dropout 0.5