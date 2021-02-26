#!/usr/bin/env python
# -*- coding:utf-8 -*-

#安装:pip install nlpcda  -i https://pypi.tuna.tsinghua.edu.cn/simple
#训练环境: Python 3.7 安装最新版本的TensorFlow和TensorFlow-gpu,如果报错说
#找一个支持CUDA10.0的Tensorflow版本。 试试tf2.2+（Keras requires TensorFlow 2.2 or higher.）
#conda activate data_augu
#下载chinese_simbert_L-12_H-768_A-12后,(| SimBERT Base  |  2200万相似句组 | 13685  | 344MB | [百度网盘](https://pan.baidu.com/s/1uGfQmX1Kxcv_cXTVsvxTsQ)(6xhq) |),需要修改config 里面的model_path
#输入是原始要数据增强的txt文件,每行是一句话,第二列是原先的label
#输出是扩增后的话
#conda activate data_aug
#conda install jieba
#conda install requests
#pip install bert4keras -i https://pypi.tuna.tsinghua.edu.cn/simple
#conda install tensorflow-gpu=2.1
#最重要的是TensorFlow版本,2.2以上会出现Keras模块找不到,2.0以下么有Keras,还需要考虑本机CuDNN的版本和TF之间的关系.
#tf2.1报错 type object 'AutoRegressiveDecoder' has no attribute 'set_rtype'
#tf2.2 报错 type object 'AutoRegressiveDecoder' has no attribute 'set_rtype'2
#运行示例:
#       export CUDA_VISIBLE_DEVICES=3
#       cd /data/public/wanghao/code/nlpcda
#       python run_SimBERT.py --input_filepath=/data/public/wanghao/code/ai-quality-violations-train/code-format-bert/data/xqb_cash_value/xqb_cash_value_merge_train.tsv --output_filepath=/data/public/wanghao/data/output_data/xqb_cash_value_merge_train_24647Lines_augumented_SimBERT.tsv


#少量测试：
#conda activate data_aug
#export CUDA_VISIBLE_DEVICES=3
#python /data/public/wanghao/code/nlpcda/run_SimBERT.py --input_filepath=/data/public/wanghao/data/xqb_die_for_augu_test_wh/xqb_die_merge_3lines_for_augument.txt --output_filepath=/data/public/wanghao/data/xqb_die_for_augu_test_wh/xqb_die_merge_3lines_for_augument_SimBERTaugumented.txt

from nlpcda import Simbert

#在bash中传参
import argparse
import os
ap = argparse.ArgumentParser()
ap.add_argument("--input_filepath", required=True, type=str, help="原始数据的输入文件目录") #输入文档路径,包含文件名
ap.add_argument("--output_filepath", required=False, type=str, help="增强数据后的输出文件目录")#输出文档路径和文件名,默认为和输入是同一个目录.
ap.add_argument("--NUM_DATA_AUGU", required=False, type=int, help="每条原始语句增强的语句数")#不输入的话,默认为3
args = ap.parse_args()



#参数处理
if args.input_filepath is None:
        print('请输入要增强的文件,注意文件第一列为句子,第二列为标签.')
        os._exit()
else:
        inputfile = args.input_filepath

if args.output_filepath:
    outputfile = args.output_filepath
else:
    from os.path import dirname, basename, join
    outputfile = join(dirname(args.input_filepath) + '/OutputAugumentedData/AugumentedWithSimBERT_' + basename(args.input_filepath))

if args.NUM_DATA_AUGU:
    num_data_augu = args.NUM_DATA_AUGU
else:
        num_data_augu = 3
#######################


IS_DEL_SimBERT_Similarity_value = 0#是否删除最后一列 衡量相似度的.

SENTENCE_MAX_LENTH = None #每个句子的最大长度,这个数字要问上游的切分句子的人他们是怎么定的.


def gen_augudata_SimBERT(inputfile, outputfile, num_data_augu):

    writer = open(outputfile, 'w')
    lines = open(inputfile, 'r').readlines()

    print("正在使用SimBert生成增强语句, 请耐心等待!")

    config = {
        'model_path': '/data/public/wanghao/code/nlpcda/model/chinese_simbert_L-12_H-768_A-12',
        'device': 'cuda',
        'max_len': SENTENCE_MAX_LENTH,
        'seed': 1
        }
    
    #初始化SimBert类
    simbert = Simbert(config=config)
    aug_sentences_all = [] #存储所有生成语句
    similarity_all = []
    label_all =[]


    # enumerate() (枚举) .函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
    for i_idx, line in enumerate(lines): #前面的枚举数字用不到,后面的line包含sentence和 tag. 对应于训练数据的格式.
        parts = line[:-1].split('\t')    #使用[:-1]是把\n去掉了
        sentence = parts[0]#前面是句子
        label = str(parts[1])#后面是之前的label
        
        if label == '2':#如果是负例,就不增强了
            continue
        
        #原文及其标签加入2个列表
        aug_sentences_all.append(str(sentence))
        similarity_all.append(str(1))
        label_all.append(str(label))
        
        #获得增强语句及其相似度        
        aug_sentences = simbert.replace(sent = sentence, create_num = num_data_augu) #得到的是一个元组,第一个是句子,第二个是相似度
        #增强语句及其相似度增加到列表中
        for aug_sentence in aug_sentences:
            aug_sentences_all.append(str(aug_sentence[0]))
            similarity_all.append(str(aug_sentence[1]))
            label_all.append(str(label))
        
        #if i_idx > 100:#做测试的时候只转换100行
        #    break


    #写入
    if IS_DEL_SimBERT_Similarity_value == 1:#是否删除最后一列相似度的部分,这样输出的文件就可以直接拿来做训练数据了.
        for i in range(len(label_all)):
            #这里生成的aug_sentence类型是tuple,有两个,要转换类型
            #print(type(aug_sentence))
            #print(aug_sentence)
            writer.write("{}\t{}\n".format(aug_sentences_all[i], label_all[i]))         #第一列 文字,第二列 原先的label, 第三列,和原来的相似度.
    else:
        for i in range(len(label_all)):
            writer.write(aug_sentences_all[i] + '\t' + label_all[i]  + '\t' + similarity_all[i] + '\n') #第一列 文字,第二列 原先的label.
            

    writer.close()
    print('All augument Done! Saved to: \n' + outputfile)


gen_augudata_SimBERT(inputfile, outputfile, num_data_augu)
#gen_augudata_SimBERT(inputfile, outputfile, num_data_augu, linerange)