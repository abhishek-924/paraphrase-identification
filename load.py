use_cuda = torch.cuda.is_available()

data_file = "../input/finaldataset1/pretrain1.tsv"
data_test_file = "../input/finaldataset1/pretest1.tsv"
training_ratio = 0.9
max_len = 30
tracking_pair = False
hidden_size = 50
batch_size = 1
num_iters = 50
learning_rate = 0.03

"""# DATA"""

data = Data(data_file,data_test_file,training_ratio,max_len)

print(len(data.word2index))

"""# Embeddings"""

embd_file = "../input/preprocess1/glove.6B.100d.txt"

#from embedding_helper2 import Get_Embedding

embedding = Get_Embedding(embd_file, data.word2index)
embedding_size = embedding.embedding_matrix.shape[1]

print(embedding_size)

len(embedding.embedding_matrix)
