import json
import logging
from tqdm import tqdm
import os
import random


def format_data():
    """
    处理诗词的json文件，构建测试集与训练集
    """

    def read_file(path):

        """
        读取json文件中1000首数据
        :return content， title
        """
        content = []
        title = []
        with open(path, encoding='utf8') as f:
            data = json.load(f)
            for item in data:
                para = ''.join(item["paragraphs"])
                tit= item["title"]

                content.append(para)
                title.append(tit)
            
        return content, title
    
    def make_data(path, start,end,start_,end_):
        with open(path,'w',encoding='utf8') as f :
            for i in tqdm(range(start,end,1000),ncols=80, desc= " ## make poet song data"):
                file_path = f"./全唐诗/poet.song.{i}.json"
                content, title= read_file(file_path)
                for con, tit in zip(content, title):
                    f.write(con+ "_!_" + tit + '\n')
            
            for i in tqdm(range(start_,end_,1000), ncols=80, desc="## make poet tang data"):
                file_path = f"./全唐诗/poet.tang.{i}.json"
                content, title= read_file(file_path)
                for con, tit in zip(content, title):
                    f.write(con+ "_!_" + tit + '\n')
    make_data("title_train_txt",0,200000,0,40000)
    make_data("title_test_txt",200000,220000,40000,48000)
    make_data("title_val_txt",220000,240000,48000,57000)

if __name__ == "__main__":
    format_data()

    