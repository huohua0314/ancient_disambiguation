import json
from collections import defaultdict
from tqdm import tqdm


def format_data() :

    def read_file(path=None):
        """
        读取json 文件中的数据
        :param path
        :return 二维list
        """

        paras = defaultdict(list)

        with open(path, encoding='utf-8') as f :
            data =  json.load(f)
            for item in data:
                idx = item['idx']
                temp = item['paragraph']
                paras[idx].append(temp)
            
            return paras
    
    def make_data(path,start, end):
        with open(path,'w' , encoding= 'utf-8' ) as f:
            for i in tqdm(range(start, end, 1000), ncols=80, desc=" ## 正在制作训练数据"):
                path = f'{(i // 1000 ) + 1 }.json'
                paras = read_file(path)

                z = 0
                for value in paras.values():
                    z = z + 1
                    if z > 10 :
                        break
                    f.write(",".join(value) + '\n')
    
    make_data("daizhi.train.txt",0, 10000)
    make_data("daizhi.valid.txt",10001, 12000)
    make_data("daizhi.test.txt", 12001, 14185)

if __name__ == "__main__" :
    format_data()

