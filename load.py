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
def commonWords(sen_1, sen_2):
  d = np.empty(len(data.word2index), dtype=int)
  for i in range(len(d)):
    d[i] = -1
    
  flag = False
    
  listPairs = []
  list1 = []
  list2 = []
  for i in range(len(sen_1)):
    d[sen_1[i]] = i
    
  for i in range(len(sen_2)):
    if d[sen_2[i]] > 1 and sen_2[i] > 0 :
      list1.append(d[sen_2[i]])
      list2.append(i)
      flag = True
      
    
  list1 = list(dict.fromkeys(list1))
  list2 = list(dict.fromkeys(list2))
  
  listPairs.append(list1)
  listPairs.append(list2)
  return listPairs

def max_pool(e_list):
  e_list = np.array(e_list)
  
  for i in range(len(e_list)):
    e_list[i] = e_list[i].data.cpu().numpy()
  mp = []
  for i in range(100):
    m = e_list[0][i]
    for j in range(len(e_list)):
      m = max(m, e_list[j][i])
    mp.append(m)
      
  #print("Length of mp = " + str(len(mp)))
  return torch.cuda.FloatTensor(mp)
