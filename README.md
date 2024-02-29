现有模型多用于处理现代汉语，对古汉语处理能力较差

本项目希望通过以多个预训练任务方式训练bert模型，增强模型处理古汉语歧义的问题

模型在 https://github.com/moon-hotel/BertWithPretrained 基础上进行更改




剩余解决问题：
- 预训练任务三数据准备
- 不同预训练任务数据整合
- 测试效果
- 预训练完成后测试最后效果

前两个预训练任务 在 ./Task 中运行 python TaskForPretraining.py 查看效果


