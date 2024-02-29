import os  
import re  
  
def traverse_files(path):
    #root当前正在遍历的目录的路径
    #dirs当前正在遍历的目录的路径
    #files当前目录下的非目录文件名列表
    
    clean_path = os.path.dirname(path)
    clean_path = os.path.join(clean_path,"clean_data")
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r+', encoding='utf-8') as f:
                try:
                    content = f.read()
                    content = [line for line in content if line.strip()] 
                    content = ','.join(map(str, content)) 
                    pattern = r'[^\w\s]'  
                    cleaned_text = re.sub(pattern, '', content) 
                    new_path = os.path.join(clean_path,file)
                    if not os.path.exists(os.path.dirname(new_path)):  
                        os.makedirs(os.path.dirname(new_path))           
                    t = open(new_path,'w')
                    t.write(cleaned_text)
                    f.close()
                    t.close()
                except:
                    pass


path = os.path.dirname(__file__)
data_path = os.path.join(path, "daizhigev20")
traverse_files(data_path)