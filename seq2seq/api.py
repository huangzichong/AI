from distutils.version import LooseVersion
import tensorflow as tf
import numpy as np
import jieba
import matplotlib.pyplot as plt
from tensorflow.python.layers.core import Dense
import pymysql.cursors

from wsgiref.simple_server import make_server
import myapp
import request
import urllib
import json
from urllib.parse import unquote,quote


with open('data/letters_source.txt', 'r', encoding='utf-8') as f:
     source_data = f.read()

with open('data/letters_target.txt', 'r', encoding='utf-8') as f:
     target_data = f.read()

connect = pymysql.Connect(
        host='localhost',
        port=3306,
        user='root',
        passwd='a86746148',
        db='lucky',
        charset='utf8'
        )

cursor = connect.cursor()
batch_size=2


    
def get_query(string):
    array = list(set(string.split('&')))
    if len(array):
       return {v.split('=')[0]:v.split('=')[1] for v in array if not(v.split('=')[0] is None)}
    else:
       return 1



def source_to_seq(text):
    '''
    对源数据进行转换
    '''
    sequence_length = len(text)
    return [source_letter_to_int.get(word, source_letter_to_int['<UNK>']) for word in " ".join(jieba.cut(text)).split(' ')] +[source_letter_to_int['<PAD>']]*(sequence_length-len(text))




def extract_character_vocab(data):
    '''
    构造映射表
    '''
    
    set_words = list(set([character for line in data.split('\n') for character in " ".join(jieba.cut(line)).split(' ')]))
    for vocab in set_words:
        sql = "select id,name from cidian where name='%s'"
        data = (vocab)
        count = cursor.execute(sql % data)
        if count == 0:
            sql = "INSERT INTO cidian (name) VALUES ( '%s' )"
            count = cursor.execute(sql % data)
            
    sql = "select id,name from cidian"
    count = cursor.execute(sql)
    result = cursor.fetchall()
    int_to_vocab = dict(result)
 
    # 这里要把四个特殊字符添加进词典
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int


# 构造映射表
source_int_to_letter, source_letter_to_int = extract_character_vocab(source_data)
target_int_to_letter, target_letter_to_int = extract_character_vocab(target_data)







def myapp(environ, start_response):
    status = '200 OK'
    headers = [('Content-type', 'text/html')]
    a=(get_query(environ['QUERY_STRING']))

    # 输入一个单词
    input_word = unquote(a['question'])
    text = source_to_seq(input_word)
    print(text)
    checkpoint = "./trained_modelTest.ckpt"

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # 加载模型
        loader = tf.train.import_meta_graph(checkpoint + '.meta')
        loader.restore(sess, checkpoint)

        input_data = loaded_graph.get_tensor_by_name('inputs:0')

        logits = loaded_graph.get_tensor_by_name('predictions:0')
        source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
        target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
        
        answer_logits = sess.run(logits, {input_data: [text]*batch_size, 
                                          target_sequence_length: [20]*batch_size, 
                                          source_sequence_length: [len(text)]*batch_size})[0] 


        pad = source_letter_to_int["<PAD>"] 
        start_response(status, headers)
        result = json.dumps(a)
        return ['{}'.format(" ".join([target_int_to_letter[i] for i in answer_logits if i != pad])).encode('utf8')]


httpd = make_server('', 5001, myapp)
print("Serving HTTP on port 5001...")
httpd.serve_forever()

