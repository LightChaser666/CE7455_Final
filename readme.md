## Source Code Structure

python 3.7+, pytorch version: 1.5.1

- /data/ contains preprocessed dataset
- glove_embedding.py is the dataset prerprocessing script.
  - **We have completed preprocessing, so you don't need to run it**
- dataset.py contains the custom dataset iterator
- model.py contains the 4 model implementation:
  - 0 - NetAB - CNN+CNN
  - 1 - NetTrans - Transformer + Transformer
  - 2 - NetTransAB - CNN + Transformer
  - 3 - NetTransBA - Transformer + CNN
- main.py is the core training script.

## Reproduction Instruction

- To run the script, you must first clone the code & data from the git, then **create an empty "/ckpt/" folder**

- Then you can reproduce all the result in the assignment with the following command:

```shell
python main.py --model 0 --lr 0.001
python main.py --model 1
python main.py --model 2
python main.py --model 3
```

- The default dataset is 'movie'. You can modify it by --dataset parameter, e.g.:

```shell
python main.py --model 1 --dataset restaurant
```

- You can choose among movie, laptop, restaurant. 
- More options, please refer to the main.py.

===========================