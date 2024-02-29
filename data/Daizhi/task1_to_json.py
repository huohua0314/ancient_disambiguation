import os
from pandas.io import json
from tqdm import tqdm
import random

def main():
    path = os.path.dirname(__file__) 
    data_path = os.path.join(path,"clean_data")
    save_path = ''
    global long
    long = 512
    global idx
    idx = 0
    global count
    count = 0
    filename = os.listdir(data_path) #获取path路径下的所有文件的名字(eg:123.txt)
    filejson = list()
    for fn in tqdm(filename):#显示进度条
        p=os.path.join(data_path,fn)
        try:
            # 大多数文件都是utf-8格式的，少数文件是gbk格式，默认使用utf-8格式读取，为了防止gbk文件使程序中断,使用try catch处理特殊情况
            f = open(p,mode="r",encoding="utf-8")
            data=f.read()

            index = 0
            while index < len(data):
                long = random.randint(30,60)
                filejson.append({'idx':idx,'paragraph':data[index : index + long]})
                index += long
            f.close()
            if count % 1000 == 999:
                save_path = os.path.join(path,   str((count + 1)//1000 ) + '.json')
                with open(save_path, 'a', encoding='utf-8') as t:
                    t.write(json.dumps(filejson, ensure_ascii=False,indent = 2))
                    t.close()
                filejson.clear()

        except Exception:
            f = open(p, mode="r", encoding="gbk")
            data=f.read()
            index = 0
            while index < len(data):
                long = random.randint(30,60)
                filejson.append({'idx':idx,'paragraph':data[index : index + long]})
                index += long
            f.close()
            if count % 1000 == 999:
                save_path = os.path.join(path,   str((count + 1)//1000 ) + '.json')
                with open(save_path, 'a', encoding='utf-8') as t:
                    t.write(json.dumps(filejson, ensure_ascii=False, indent = 2))
                    t.close()
                
                filejson.clear()
        idx += 1
        count += 1
    save_path = os.path.join(path,   str((count )//1000  + 1) + '.json')
    with open(save_path, 'a', encoding='utf-8') as t:
        t.write(json.dumps(filejson, ensure_ascii=False, indent = 2))
        t.close()
    return 0

main()