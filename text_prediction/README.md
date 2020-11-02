# Text Prediction

I wanted to explore text prediction using a neural network as a way to generate
search suggestions for users of the pgAdmin and PostgreSQL websites. There are
two scripts in this directory:

## html-train.py

This script will take a directory of HTML files and train a model based on the
text within those files. It will save the model and the tokenizer that contains
the data.

```shell script
(ml) dpage@hal:~/git/machine_learning/text_prediction$ python3 html-train.py --help
usage: html-train.py [-h] [--debug] -d DATA -i INPUT -m MODEL

Create a Tensorflow model based on a directory of HTML

optional arguments:
  -h, --help            show this help message and exit
  --debug               enable debug mode
  -d DATA, --data DATA  the file to save data to
  -i INPUT, --input INPUT
                        the input directory containing the HTML files
  -m MODEL, --model MODEL
                        the file to save the model to
```

## test-model.py

This script will load the model and tokenizer created during training, and 
allow you to enter words and then select the number of additional words to 
predict.

```shell script
(ml) dpage@hal:~/git/machine_learning/text_prediction$ python test-model.py  --help
usage: test-model.py [-h] -d DATA -m MODEL

Test a pre-trained Tensorflow model with text data

optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  the file to load data from
  -m MODEL, --model MODEL
                        the file to load the model from
```

For example:

```shell script
(ml) dpage@hal:~/git/machine_learning/text_prediction$ python test-model.py -d pgadmin-docs.json -m pgadmin-docs.h5 
Enter text (blank to quit): trigger
Number of words to generate (default: 1): 
Results: trigger date
Enter text (blank to quit): table
Number of words to generate (default: 1): 3
Results: table you can be
Enter text (blank to quit): 
```