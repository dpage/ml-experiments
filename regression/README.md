# Regression

We can use Tensorflow from within PostgreSQL to perform regression tasks for
modelling and prediction. Essentially we're using a neural network to learn the
relationship between inputs and outputs; for example, given:

x = y *[some operation]* z

we're modelling *[some operation]*, essentially approximating the formula.

Note that whilst in some cases there may be a specific, defined relationship
between the inputs and outputs (e.g. x<sup>2</sup> = y<sup>2</sup> + 
z<sup>2</sup> - Pythagoras' theorem), in other cases there may not be. These
are typically the cases that interest us as they allow us to analyse data for
business intelligence purposes. A good example is predicting the price of a 
house based on factors such as location, number of bedrooms and reception rooms,
type of build etc.

## PostgreSQL

We need to configure PostgreSQL in order to run Tensorflow. This consists of a
couple of steps:

1. Install pl/plython3 in your PostgreSQL database, e.g:

    ```postgresql
    CREATE EXTENSION plpython3u;
    ```
   
2. Install Tensorflow (and any other required modules) in the Python environment
used by the PostgreSQL server. In my case, that's the EDB LanguagePack on
macOS:

    ```shell script
    % sudo /Library/edb/languagepack/v1/Python-3.7/bin/pip3 install tensorflow numpy
   ```
   
It should then be possible to create pl/python3 functions in PostgreSQL.

## Scripts

There are various SQL scripts in this directory that represent the various
experiments I've worked on. Most create:

- A table to store training inputs and outputs.
- A function to generate data for the training table.
- A function to train a model and make a prediction based on that model.

Obviously in real-world applications the model creation and prediction functions
would probably be separated, and training data would likely come from existing
tables/views.

__Note:__ All files written by the scripts will be owned by the user account
under which PostgreSQL is running, and unless a full path is given, will be 
written relative to the data directory. I've used */Users/Shared/tf* as the 
working directory; you may want to change that.

### ohms_law.sql

This attempts to teach a network a basic operation based on Ohms Law;

voltage (v) = i (current) * r (resistance)

It's worth noting that the results of this model are *terrible*, so don't try 
to use it as the basis for anything else. At the time of writing I haven't yet
figured out why this is the case, though I have some hunches.

### pythagoras.sql

This attempts to teach a network Pythagoras' Theorem:

x<sup>2</sup> = y<sup>2</sup> + z<sup>2</sup>

The square of the length of the hypotenuse of a right angled triangle is the 
sum of the square of the other sides.

### random1.sql

This attempts to teach a network a completely fictitious equation with five 
input variables (in pl/pgsql):

 z := cbrt((a * b) / (sin(c) * sqrt(d)) + (e * e * e));
 
 ### housing.sql
 
 This is based on the well known [Boston Housing dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html).
 The SQL file contains the definition for a table to hold the data (loading it
 is an exercise for the reader) and a function for training, testing and using
 a model. This function differs from other in that Pandas data frames are used
 in place of Numpy, and an attempt is made to remove rows container outliers 
 from the dataset before training, in order to increase accuracy of results.