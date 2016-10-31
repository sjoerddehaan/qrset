import os
import numpy as np
import string
import pyqrcode
import pandas as pd
from scipy import misc
from skimage.transform import resize

ALPHABET = string.ascii_letters + string.punctuation

def random_messages(n, length, alphabet, variable_length=True):
    """ Returns list of n random messages of length l,
        by drawing characters from alphabet at random
    """
    if variable_length:
        remain = n % 2
        n = n // 2 + remain
    alphabet = np.array(list(alphabet))
    characters = np.random.choice(alphabet, [n, length])
    strings = ["".join(characters[i]) for i in range(n)]
    if variable_length:
        messages = []
        split = np.random.randint(length, size=[n])
        for i, s in enumerate(split):
            messages.append(strings[i][:s])
            messages.append(strings[i][s:])
        messages = messages[: 2*n - remain]
    else:
        messages = strings
    return messages


def generate_messages(n, length, output_dir, alphabet=ALPHABET, variable_length=True):
    """ Adds random string messages to an indexed list,
        kept in output_dir/y.csv.
        Outputs two pandas dataframes: updated indexed
        list, and indexed list with new messages
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    y_file = output_dir + '/y.csv'
    if not os.path.exists(y_file):
        y = pd.DataFrame({'y': []})
        y.index.name = 'Index'
        index = np.arange(n)
    else:
        y = pd.read_csv(y_file, index_col=0, sep="\t")
        start = y.index.max()
        index = np.arange(start + 1, start + n + 1)
    messages = random_messages(n, length, alphabet, variable_length=True)
    y_add = pd.DataFrame({'y': messages}, index=index)
    y_add.index.name = 'Index'
    y = y.append(y_add) 
    y.index.name = 'Index'
    y.to_csv(y_file, sep='\t')
    return y, y_add


def generate_png(y, output_dir):
    """ Takes an indexed list of messages
        Generates png in output_dir
        for each message
    """
    for i, message in y.itertuples():
        output_file = output_dir + '/x_{}.png'.format(i)
        code = pyqrcode.create(message)
        code.png(output_file, scale=8)
        
def generate_messages_png(n, length, output_dir, variable_length=True):
    """ Generates an indexed list with messages in
        output_dir/y.csv.
        Generates png imges with QR codes of the messages
        in output_dir/x_{index}.jpg.
    """
    _, y_new = generate_messages(n, length, output_dir, ALPHABET, variable_length)
    generate_png(y_new, output_dir) 


def load_qr_set(input_dir):
    """ Load dataset of QR images in png format
        from disk.
    """
    y = pd.read_csv(input_dir + '/y.csv', sep="\t")
    codes = {}
    for i in y.index:
        codes[i] = misc.imread(input_dir + '/x_{}.png'.format(i))
    x = pd.DataFrame.from_dict(codes, orient='index')
    x.columns=['x']
    dataset = y.join(x, how='inner').drop('Index', axis=1).head()
    return dataset


def generate_qr_array(message, image_shape=(100, 100)):
    """ Returns a numpy array image representation
        of a QR code.
    """
    code = pyqrcode.create('text')
    lines = code.text().split('\n')
    lines = lines[:-1] # Remove empty line
    a = np.concatenate([np.array(list(line), dtype=np.uint8) for line in lines])
    a = 255 * (1 - a)
    shape = (len(lines), len(lines[0]))
    a = a.reshape(shape)
    a = resize(a, image_shape)
    return a




