"""
pandas分批读入标注数据,将其转化为[data1,data2,...],82开划分测试集,训练集
经过BertEmbedding与Bert整个模型后将训练集语料转化为词向量后取平均值,得到训练完的词表向量
而后进行测试集
"""

from model import BertEmbedding
from model import Bert
from model import BertConfig
from model import BertForPretrainingModel
from utils import Vocab
import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
class data:
    def __init__(self,Masked_Position,Word_id,Sense_id,text,meaning):
        self.Masked_Position = Masked_Position #词语的位置,由于语料中该词可能不止出现一次,因此用list对象进行存储
        self.Word_id = Word_id #词语的id
        self.Sense_id = Sense_id #义项的id
        self.content = text #语料
        self.meaning = meaning #义项描述

class Vocab:
    """
    根据本地的vocab文件，构造一个词表
    vocab = Vocab()
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    print(vocab.stoi['我'])  # 通过单词返回得到词表中对应的索引
    print(len(vocab))  # 返回词表长度
    """
    UNK = '[UNK]'

    def __init__(self, vocab_path):
        self.stoi = {}
        self.itos = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, word in enumerate(f):
                w = word.strip('\n')
                self.stoi[w] = i
                self.itos.append(w)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __len__(self):
        return len(self.itos)

def build_vocab(vocab_path):
    """
    vocab = Vocab()
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    print(vocab.stoi['我'])  # 通过单词返回得到词表中对应的索引
    """
    return Vocab(vocab_path)

def parse_masked_string(masked_string,word_id,sense_id,meaning):
    """
    Input:语料
    去除掉语料中的空格并生成data对象
    Output:data实例
    """
    clean_string = masked_string.replace('[', '').replace(']', '')

    my_string = masked_string
    sub_string = '['
    positions = []
    start_pos = 0
    while True:  
        found_pos = my_string.find(sub_string, start_pos)  
        if found_pos == -1:
            break
        positions.append(found_pos)  
        start_pos = found_pos + len(sub_string)  # 从下一个位置开始继续查找 

    return data(positions,word_id,sense_id,clean_string,meaning)

def read_data(path):
    """
    读入路径
    return [data1,data2,...]
    """
    # chunksize = 1000  # 每次读取的行数  
    # for chunk in pd.read_excel('1113_icip_ancient_chinese_annotation_corpus.xlsx', chunksize=chunksize): 
    #     pass
    # content = pd.read_excel(path)
    content = pd.read_excel(path,sheet_name='语料库')
    result = list()
    for _,row in content.iterrows():
        result.append(parse_masked_string(row['语料'],row['词语id'],row['义项id'],row['义项描述']))

    return result

def read_vocab(path):
    """
    读入词表
    Input: path
    Output: 含有索引的dict
    """
    # 初始化一个空字典来存储词和它们的索引  
    vocab_dict = {}  
    # 打开文件  
    with open(path, 'r', encoding='utf-8') as file:  
        # 遍历文件的每一行  
        for index, line in enumerate(file):  
            # 去除每行末尾的换行符，并将词作为键，索引作为值存入字典  
            word = line.strip()  
            vocab_dict[word] = index  
    # 返回包含索引的字典  
    return vocab_dict 

def data_process(data_list):
    """
    Input: [data1,data2,...]
    实现句子和义项的分类
    Output: 将目标词Wi标注为义项Sj的句子 [[data(w1,s1),data(w1,s1),...],[data(w1,s2),data(w1,s2),...],...]
    """
    # 创建一个字典，用于存储分类后的data对象  
    # 键为(Word_id, Sense_id)的元组，值为对应的data对象列表  
    grouped_data = {}  
      
    # 遍历data_list  
    for item in data_list:  
        # 生成键  
        key = (item.Word_id, item.Sense_id)  
        # 如果键不存在于字典中，则初始化一个空列表  
        if key not in grouped_data:  
            grouped_data[key] = []  
        # 将data对象添加到对应的列表中  
        grouped_data[key].append(item)
        # 将字典的values转换为列表，这就是最终的结果  

    return list(grouped_data.values()) 

def get_token_id(data,vocab,src_len):
        """
        Input: data类,vocab词表字典,
        Output: position_id 形状为[1,str_len]
        将data实例转化为position_id词向量
        """
        content = data.content
        word_vector = list()
        for each_word in content:
            word_vector.append(vocab[each_word])
        #进行padding操作
        for i in range(src_len - len(word_vector)):
            word_vector.append(0)
        print(word_vector) #调试代码
        print(len(word_vector)) #调试代码
        return word_vector
    
def get_position_id(word_vector):
    """
    Input:词向量
    得到position_id
    Output: position_id [1,str_len]
    """
    position_id = []
    for i in range(len(word_vector)):
        position_id.append(i)
    print(position_id) #调试代码
    return position_id

def To_Vector(data,max_str_len,batch_size = 1,path="bert_base_chinese\\vocab.txt"):
    #此处batch_size暂时设置为1,即只传入一句话进行测试
    """
    input:data,最大的样本长度,批量大小
    经过embedding层和transformer块
    return: word vector
    单个字的位置怎么标出？
    """
    my_vocab = Vocab(path)
    json_file = 'bert_base_chinese/config.json'
    config = BertConfig.from_json_file(json_file)
    config.__dict__['use_torch_multi_head'] = True
    src = get_token_id(data,my_vocab)
    position_id = get_position_id(src)
    src = torch.tensor(src)
    src = src.view(max_str_len,batch_size) #将其转化为[str_len,batch_size]的torch张量
    bertembedding = Bertembedding(config)
    bert_embedding_result = bertembedding(src,position_id)
    #通过embedding层,通过BertEncoder部分如下
    return

def test():
    """
    Input: 目标词Wi标注为义项Sj的句子
    遍历列表数据进行训练
    Output: 义项的词向量表示集合[S1,S2,...]
    """
    return

def accuracy():
    """
    计算准确率
    """
    return


def test():
    """
    Input: 测试集 目标词Wi标注为义项Sj的句子
    return: 每个词判别的准确率[S1_accurace,S2_accuracy,...]
    """
    return 

def distance_compute():
    """
    计算余弦相似度
    返回每个义项的可能性
    """
    return

def stats(data_list):
    """
    对标注语料进行统计
    确定最少训练测试句数
    """
    frequency = []
    times = []
    idx = []
    for i in range(len(data_list)):
        #统计每个词每个义项的标注数量
        frequency.append(len(data_list[i]))
        
    for i in range(2,11):
        count = 0
        for each in frequency:
            if each >= i:
                count += 1
        times.append(count / len(frequency))
        idx.append(i)
    
    plt.plot(idx,times)
    plt.scatter(idx, times, c='pink', s=50, alpha=0.5) 
    plt.show()
    return 0

        
class Disambiguation:
    def __init__(self, ) -> None:
        self.model = BertForPretrainingModel




