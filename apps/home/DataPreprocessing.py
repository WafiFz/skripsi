import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

# nltk.download('stopwords')
# stop_english = stopwords.words('english')
# stop_indonesian = stopwords.words('indonesian')

def basic_preprocessing(text):
    text=text.lower()
    text=re.sub(r'[^\w\s]','',text)
    text = re.sub(r'@\w+', '', text)
    return text

def remove_duplicates(input):

    # split input string separated by space
    input = input.split(" ")

    # now create dictionary using counter method
    # which will have strings as key and their
    # frequencies as value
    UniqW = Counter(input)

    # joins two adjacent elements in iterable way
    s = " ".join(UniqW.keys())
    return s

def remove_stop(text, stop):
    result = ' '.join([word for word in text.split() if word not in stop])
    return result


def text_preprocessing(text, is_indonesia):
    text = basic_preprocessing(text)
    # text = remove_duplicates(text)

    # stop_words = stop_indonesian if is_indonesia else stop_english 
    # text = remove_stop(text, stop_english)

    return text
