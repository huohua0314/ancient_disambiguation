**clean_data** 文件夹中数据来源为殆知库数据经去除标点空白符后所得
数据来源：https://github.com/garychowcmu/daizhigev20

下载数据
运行:  python task1_data_clean.py 去除标点与空格
运行： python task1_to_json.py 将文本数据每隔30-60字符分割为一段，生成json文件用于后续使用
运行： python read.py 处理json文件，获得实验所需训练集和测试集



