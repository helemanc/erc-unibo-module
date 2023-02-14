import re 
import string


def remove_digits(input_text):
    """Removes any digits from the text."""
    return re.sub(r'\d+', ' ', input_text)


def remove_punctuation(input_text):
    """Replaces any punctuation symbol by a space."""
    punct = string.punctuation + '…“”￼＆—●–'
    punct = str(set(punct)-set(['!','?']))
    trantab = str.maketrans(punct, len(punct)*' ')
    return input_text.translate(trantab) 


# def add_space(input_text, consider_marks = True):
#     if consider_marks:
#     input_text = input_text.replace('?', ' ? ')
#     input_text = input_text.replace('!', ' ! ')
#     else:
#     input_text = input_text.replace('?','')
#     input_text = input_text.replace('!','')
#     return input_text



def to_lower(input_text):
    """ Transform given text in lower case. """
    return input_text.lower()


def apostrophe(input_text):
    """ Substitute '’' with standard apostrophe "'". """
    return re.sub("(\s?)([′’‘‘'])(\s?)", "'", input_text) 


def spaced_punctuation(input_text):
    return re.sub("(\s)([\.!#$%&()*+,-/:;<=>\?@\[\]^_{|}~])(\s?)", '\g<2> ',input_text)


def double_quotes(input_text):
    return re.sub('[“”]','"',input_text)

def sus_dots(input_text):
    return re.sub('[…]','...',input_text)

def comma(input_text):
    return re.sub('[、]',', ',input_text)

def dot(input_text):
    return re.sub('[。]','.',input_text)

def dashes(input_text):
    return re.sub('[-–—]','-',input_text)


def remove_format(input_text):
    """ Transform formatted text in standard text 
        (i.e. using only target letters)
    """
    return format(input_text)

def strip_text(input_text):
    """ Removes spaces at the beginning and at the end of the string."""
    return input_text.strip()

def remove_multi_space(input_text):
    return str.join(' ', input_text.split())


def preprocessing_pipeline(input_text):
    """ Applies all the preprocessing steps to the input text. """
    input_text = apostrophe(input_text)
    input_text = double_quotes(input_text)

    input_text = comma(input_text)
    input_text = dot(input_text)
    input_text = sus_dots(input_text)
    input_text = dashes(input_text)
    input_text = spaced_punctuation(input_text)
    input_text = strip_text(input_text)
    return input_text