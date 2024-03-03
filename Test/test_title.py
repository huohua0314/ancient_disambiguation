import sys

sys.path.append('../')
from utils.log_helper import logger_init
import logging
from transformers import BertTokenizer
import os
from utils.create_pretraining_data import LoadTitleMatchDataset


class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # daizhiku 数据集配置
        self.dataset_dir = os.path.join(self.project_dir,'data', 'title')
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_chinese")
        self.train_file_path = os.path.join(self.dataset_dir, 'title.train.txt')
        self.val_file_path = os.path.join(self.dataset_dir, 'title.valid.txt')
        self.test_file_path = os.path.join(self.dataset_dir, 'title.test.txt')

        self.data_name = 'title'



        self.seps = "_!_"

        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')

        self.is_sample_shuffle = True
        self.batch_size = 16
        self.max_sen_len = None
        self.max_len = 400

        self.max_position_embeddings = 512
        self.pad_index = 0
        self.is_sample_shuffle = True

        self.match_rate = 0.5
        self.log_level = logging.DEBUG

        logger_init(log_file_name=self.data_name, log_level=self.log_level,
                    log_dir=self.logs_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        # 将当前配置打印到日志文件中
        logging.info(" ### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")

if __name__ == '__main__':
    model_config = ModelConfig()
    data_loadeer = LoadTitleMatchDataset(vocab_path=model_config.vocab_path,
                                         tokenizer=BertTokenizer.from_pretrained(
                                                 model_config.pretrained_model_dir).tokenize,
                                        batch_size=model_config.batch_size,
                                        max_position_embeddings=model_config.max_position_embeddings,
                                        max_len=model_config.max_len,
                                        max_sen_len=model_config.max_sen_len,
                                        pad_index=model_config.pad_index,
                                        is_sample_shuffle=model_config.is_sample_shuffle,
                                        random_state=model_config.match_rate,
                                        match_rate=model_config.match_rate
                                         )
    test_iter = data_loadeer.load_train_val_test_data(True,test_file_path=model_config.test_file_path)
    for b_token_ids, b_segs, b_match_label in test_iter:
        print(b_token_ids.shpae)
        print(b_segs.shape)
        print(b_match_label.shape)

        break