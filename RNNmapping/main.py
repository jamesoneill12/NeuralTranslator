from gensim.models import Word2Vec, Phrases
import pickle
from helpers import clean_str
from ANN import RNN,prepare_data,embedding_format

root = "C:/Users/1/James/grctc/GRCTC_Project/Classification/"
write_path = root + "Sequential_Models/word2vector/"
filename = root + 'Preprocessing/data/' \
                  '' \
                  'FinalAnnotationsModality_sentences_wArtificialProhibitions.txt'
# "Preprocessing/data/FinalAnnotationsModality_sentences.txt"

googleVecs = "C:/Users/1/James/grctc/GRCTC_Project/Classification/Data/Embeddings/word2vec/GoogleNews-vectors-negative300.bin"
file = '/annotated_data/EU.AML2015_new.txt'
rest_path = "C:/Users/1/James\REST/minimal-django-file-upload-example/src/" \
            "for_django_1-9/myproject/myproject/test/vectors/"

sentences = [clean_str(line.decode('utf-8').strip()).split() for line in open(filename, "r").readlines()]
legal_sentences = pickle.load(open(root + "/XMLParsers/eurolex_documents.pkl"))
#X = Word2Vec(legal_sentences, size=100, window=5, min_count=5, workers=4)
X = Word2Vec.load_word2vec_format(googleVecs, binary=True)  # C binary format

# test data prep is correct
root = "C:/Users/1/James/grctc/GRCTC_Project/Classification/Word2Vec/annotated_data/"
emb,y = prepare_data(filename = root+'EU.AML2015_new.txt')
#emb = embedding_format(emb)
print (emb.shape)
model = RNN(X=emb,y=y,h_dim=5,num_class=3,type='lstm',pad=100)