import os 
import sys
sys.path.append('./')
from model import BertForPretrainingModel
from model.BasicBert.BertConfig import BertConfig
import pandas as pd
import numpy as np
import logging
import pretty_errors
import random
import torch
import tqdm
import more_itertools as mit  
import torch.nn.functional as F  
from matplotlib import pyplot as plt

class data:
    def __init__(self,Masked_Position,Word_id,Sense_id,text,meaning):
        self.Masked_Position = Masked_Position #词语的位置,由于语料中该词可能不止出现一次,因此用list对象进行存储
        self.Word_id = Word_id #词语的id
        self.Sense_id = Sense_id #义项的id
        self.content = text #语料
        self.meaning = meaning #义项描述
        self.hidden_vector = None #词向量
    
    def Set_Hidden_Vector(self,hidden_vector) -> None:
        assert self.hidden_vector == None
        self.hidden_vector = hidden_vector

class train_res:
    def __init__(self,Word_id,Sense_id,hidden_vector) -> None:
        self.Word_id = Word_id #词语的id
        self.Sense_id = Sense_id 
        self.hidden_vector = hidden_vector
        self.test_times = None
        self.correct_times = None

    def Get_accuracy(self):
        return self.correct_times / self.test_times
    
    def Add_test_times(self) -> None:
        self.test_times += 1

    def Add_correct_times(self) -> None:
        self.correct_times += 1


class Vocab:
    """
    根据本地的vocab文件,构造一个词表
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
    
class Disambiguation:
    def __init__(self, model_path) -> None:
        self.config = ModelConfig()
        self.model = BertForPretrainingModel(self.config,model_path)

    def hidden_vector(self, input_ids,attention_mask=None,token_type_ids=None,position_ids=None):
        return self.model.hidden_vector(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )
    
class ModelConfig(object):
    def __init__(self) -> None:
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_chinese")
        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        # 将当前配置打印到日志文件中
        logging.info(" ### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")

def read_data(path):
    """
    读入路径
    return [data1,data2,...]
    """
    content = pd.read_excel(path,sheet_name='语料库')
    result = list()
    for _,row in content.iterrows():
        result.append(parse_masked_string(row['语料'],row['词语id'],row['义项id'],row['义项描述']))

    return result

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

def get_token_id(data,vocab):
        """
        Input: data类,vocab词表字典,
        Output: position_id 形状为# [src_len, batch_size]
        将data实例转化为position_id词向量
        """
        content = data.content
        word_vector = list()
        for each_word in content:
            word_vector.append(vocab[each_word])

        return torch.from_numpy(np.transpose(word_vector))

def Write_To_File(path='./Task/test.xlsx'):
    """
    将语料通过模型,得到hidden_vector
    Input: path
    Output: [data1,data2,...]
    给data对象赋值hidden_vector
    """
    model_path = './bert_base_chinese'
    vocab_path = './Task/vocab.txt'
    #my_test = Disambiguation(model_path)
    vocab = Vocab(vocab_path)
    content = read_data('./Task/1113_icip_ancient_chinese_annotation_corpus.xlsx') #读入的内容是[data1,data2,...]
    print("read_data finished!")
    test_model = Disambiguation(model_path) 
    # print(content[10].meaning)
    hidden_vector_list,Masked_Position_list = [],[]

    def batch_iterator(lst, batch_size):  
        for i in range(0, len(lst), batch_size):  
            yield lst[i:i+batch_size] 


    def batch_iterator(lst, batch_size):  
        for i in range(0, len(lst), batch_size):  
            yield lst[i:i+batch_size] 

    batch_size = 10
    cnt = 0
    for batch in batch_iterator(content,batch_size):
        print(cnt)
        for _ in range(len(batch)):
            word_vector = get_token_id(content[cnt],vocab)
            hidden_vector = test_model.hidden_vector(word_vector)
            Masked_Position = content[cnt].Masked_Position
            hidden_vector_list.append(hidden_vector)
            Masked_Position_list.append(Masked_Position)
            content[cnt].Set_Hidden_Vector(hidden_vector)
            cnt += 1
        data_list = {'hidden_vector':hidden_vector_list,
     'masked_position':Masked_Position_list}
        torch.save(data_list,'hidden_vector.pt')
        hidden_vector_list,Masked_Position_list = [],[]
        
    print("transfer to hidden vector finished!")

def Get_average_vector():
    """
    Input: None
    return: 
    训练结果 [[train_res1,train_res2],...] train_res包含word_id,sense_id,hidden_vector将word_id相同的置于一个列表之中
    测试集 test_set [data1,data2,...]
    """
    data_list = Write_To_File()
    # 确定训练集和测试集的大小  
    train_size = int(0.8 * len(data_list))  
    test_size = len(data_list) - train_size  
    # 使用random.sample随机选择训练集  
    train_set = random.sample(data_list, train_size)  
    # 从原始列表中移除训练集的数据，剩下的就是测试集  
    test_set = list(set(data_list) - set(train_set))  
    # 确保测试集大小正确  
    assert len(test_set) == test_size 

    train_set = data_process(train_set) #[[data(w1,s1),data(w1,s1),...],[data(w1,s2),data(w1,s2),...],...]
    train_res = list() #[train_res1,train_res2,...]
    for i in range(len(train_set)):
        for j in range(len(train_set[i])):
            tensor_list = []
            masked_position = train_set[i][j]
            for k in range(len(masked_position)):
                position = masked_position[k]
                tensor_list.append(train_set[i][j].hidden_vector[position])
        if tensor_list:
            stacked_tensors = torch.stack(tensor_list)
            average_tensor = torch.mean(stacked_tensors, dim=0)
            res = train_res(train_set[i][0].Word_id,train_set[i][0].Sense_id,average_tensor)
            train_res.append(res)
    
    res = list()
    for i in range(len(train_res)):
        temp = []
        if len(temp) == 0:
            temp.append(train_res[i])
        elif temp[0].Word_id == train_res[i].Word_id:
            temp.append(train_res[i])
        elif temp[0].Word_id != train_res[i].Word_id:
            res.append(temp)
        else:
            print(train_res[i].Word_id)
            print(temp[0].Word_id)
            raise ValueError('请检查train_res内容')

    return res,test_set

def test():
    """
    Input:
    train_res训练结果 [[train_res1,train_res2],...] train_res包含word_id,sense_id,hidden_vector将word_id相同的置于一个列表之中
    test_set测试集 test_set [data1,data2,...]
    return:
    准确率
    """
    def distance_compute(tensor1,tensor2):
        """
        计算余弦相似度
        返回每个义项的余弦相似度
        """
        tensor1_norm = F.normalize(tensor1, dim=0)  
        tensor2_norm = F.normalize(tensor2, dim=0) 
        cosine_similarity = torch.dot(tensor1_norm, tensor2_norm) 
        return cosine_similarity

    train_res,test_set = Get_average_vector()
    # test_list = [] # 用于存储train_res
    for i in range(len(test_set)): #遍历测试集
        for j in range(len(train_res)): #查找相同的word_id
            if test_set[i].Word_id == train_res[j][0].Word_id:
                prob_list = list()
                for k in range(len(train_res[j])): #遍历该word_id下的所有sense_id
                    if train_res[j][k].Sense_id == test_set[i].Sense_id: #在该义项的测试次数上加一
                        train_res[j][k].test_times += 1
                    prob = distance_compute(test_set[i].hidden_vector[0],train_res[j].hidden_vector) #计算余弦相似度
                    prob_list.append(prob)
                    print("在{}中,义项为{}的概率为{}".format(train_res[j].content,train_res[j].sense_id,prob))
                #得到相似性最高的义项
                max_value = max(prob_list)
                max_position = prob_list.index(max_value)
                if max_value == k:
                    train_res[j][max_position].correct_times += 1
                break
            else:
                continue
    
    # train_times = 3 #最少训练次数
    total_times,total_correct_times = 0,0
    for i in range(len(train_res)): #遍历测试集
        for j in range(len(train_res[i])): 
            total_times += train_res[i][j].test_times
            total_correct_times += train_res[i][j].correct_times
    return  total_correct_times / total_times


if __name__ == '__main__':
    correct_rate = test()
    print("整体准确率为{}".format(correct_rate))

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
