from gensim.models import KeyedVectors

dir = "E:/data/WordVectors/"
# model = KeyedVectors.load_word2vec_format(dir + "BioWordVec_PubMed_MIMICIII_d200.vec.bin", binary=True, limit=10)
model = KeyedVectors.load(dir + "tmp.model", mmap='r')
model.save_word2vec_format(dir + "bio-word-vectors.vec", binary=False)
