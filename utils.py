import oci
import numpy as np

# read a txt file


def read_file(file_path):
    '''
    Reads a text file and returns the content as a string
    '''
    with open(file_path, 'r') as file:
        return file.read()


def divide_docs(doc, cut_len=1750):
    '''
    Helper function that divides a string into chunks of token size less than 512
    '''
    doc_len = len(doc)
    if doc_len % cut_len == 0:
        n = doc_len // cut_len
    else:
        n = (doc_len // cut_len) + 1
    chunks = []
    for i in range(n):
        chunks.append(doc[i*cut_len:(i+1)*cut_len])
    return chunks


def recover_doc(df_docs, doc_title):
    '''
    Function to recover the original document from the chunks
    '''
    doc = ''
    for i in range(len(df_docs)):
        if df_docs.loc[i, 'doc_title'] == doc_title:
            doc += df_docs.loc[i, 'doc_text']
    return doc


def calculate_similarity(a, b):
    '''
    Calculates the similarty between two vectors
    '''
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
