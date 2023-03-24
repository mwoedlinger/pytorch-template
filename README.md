# Pytorch-template

A simple template for pytorch projects. The template uses [Spock](https://fidelity.github.io/spock/) for handling config files and command line arguments. The sample projects train a simple single image compression model on a dataset of images.

## Data

The datasets are specified in text files containing one sample name per row. To such a text file from an existing folder of images simply use 
```
find /path/to/folder/*.png > /path/to/data/train.txt
```
Similarly for `eval.txt` and `test.txt`. 

## Config

The valid config parameters are specified in `src/config.py`. A sample config is available in `configs/default.yaml`

## Experiment

To start training run the `train.py` file:

```
python train.py --config configs/default.yaml
```

For every experiment a new directory is created with the name of the corresponding wandb run. At the beginning of training a copy of the source code, the data *.txt files and the config is made and saved in this directory. After every evaluation the model weights as well as optimizer and learning rate scheduler are saved as well. To continue an existing expirement set the `resume` or `resume_with_new_exp` to true, the latter only loads training status (model weight, optimizer, ...) but creates a new experiment folder and wandb run.