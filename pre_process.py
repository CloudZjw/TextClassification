import jieba
import jieba.analyse
import re


# a-zA-Z0-9
def cut_sentence(sen):
    cut1 = re.sub('[’!"#$%&\'()*+,-./:;<=>?@，。?～★、…【】（）《》？“”‘’！[\\]^_`{|}~\s\n]+', '', sen)
    cut2 = re.sub(
        '[\001\002\003\004\005\006\007\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a]+',
        '', cut1)
    return cut2


# 对训练集进行分词，返回out_records: 文本内容列表
# tag_list: 文本内容对应的分类
def txt_process(filepath):
    out_records = []
    tag_list = []
    with open(filepath, 'r', encoding='UTF-8') as file:
        for line in file:
            # 提取每一行的数据
            record = line.strip('\n').split('|')
            # 每条记录格式：数据ID|标签|文本内容，这里提取文本内容进行分词
            seg_list = jieba.cut(cut_sentence(record[2]), cut_all=False)
            # 保存每条记录的标签
            tag = record[1]
            tag_list.append(tag)

            # keywords = jieba.analyse.extract_tags(record[2], topK=5, withWeight=False)
            # keywords_all.append(keywords)
            # stopwords = stopwords_list('resources//cn_stopwords.txt')

            out_record = []
            for word in seg_list:
                out_record.append(word)
            out_records.append(out_record)
    return out_records, tag_list


# 输出文本内容到文件
def write_words(records, output_filepath):
    with open(output_filepath, 'w', encoding='utf-8') as file:
        for record in records:
            for word in record:
                file.write(word + ' ')
            file.write('\n')


# 输出标签
def write_tag_list(tag_list, t_filepath):
    with open(t_filepath, 'w', encoding='utf-8') as file:
        for tag in tag_list:
            file.write(tag + '\n')


if __name__ == "__main__":
    # 处理训练集和验证集
    words_train, tag_list_train = txt_process('resources//train.txt')
    words_val, tag_list_val = txt_process('resources//val.txt')
    # 输出到文本文件
    write_words(words_train, 'resources//words_train.txt')
    write_words(words_val, 'resources//words_val.txt')
    write_tag_list(tag_list_train, 'resources//tags_train.txt')
    write_tag_list(tag_list_val, 'resources//tags_val.txt')
