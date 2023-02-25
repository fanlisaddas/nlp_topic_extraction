# coding=utf-8
from read_articles import get_articles
from gensim.models.ldamodel import LdaModel
from pprint import pprint
from lda import tokenize
from lda import press_anything_continue
import lda
import sys

"""
# 打包时使用外部输入路径
training_path = sys.argv[1]
testing_path = sys.argv[2]
"""

# 训练文本
training_path = 'data/5G_chapter4.docx'
# 测试文本
testing_path = 'data/5G_chapter4_abstract.docx'

if __name__ == '__main__':
    # 读取训练集数据
    print("读取训练文本：", training_path)
    fulltext = get_articles(training_path)
    """
    文本处理: 分词、生成字典和词袋
    dictionary: 字典
    corpus: 词袋
    corpus_tfidf: 使用TF-IDF算法处理后的词袋
    """
    dictionary, corpus, corpus_tfidf = lda.split_and_process(fulltext)
    print("已完成文本处理!")
    """
    输出文本处理后的训练语料（字典、词袋）
    """
    lda.show_dictionary_and_corpus(dictionary, corpus, corpus_tfidf)
    # 在 1~100 之间选择perplexity最小的主题数
    num_topics = lda.find_optimal_topic_num(100, dictionary, corpus_tfidf)
    print("训练lad模型采用的主题数：\n", num_topics)
    # 训练模型
    press_anything_continue()
    print("正在训练lda模型...")
    lda_model = LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary, passes=5)
    print("lda模型训练完成！")
    """
    # 加载已训练好的模型
    model_name = "./model/model.lda"
    if os.path.exists(model_name):
        lda_model = LdaModel.load(model_name)
        print("加载已有模型")
    else:
        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=id2word, passes=5)
        # 保存模型
        lda_model.save(model_name)
        print("加载新创建的模型")
    """
    # 输出训练文本中最重要的10个主题，每个主题包含6个词
    press_anything_continue()
    print("输出训练文本中最重要的10个主题，每个主题包含6个词：")
    pprint(lda_model.print_topics(10, 7))
    # 模型检验
    press_anything_continue()
    print("模型检验：\n")
    print("读取测试文本：", testing_path)
    unseen_document = get_articles(testing_path)
    for para in unseen_document:
        print("测试文本的内容如下:", para)
        print("\n")
        bow_vector = dictionary.doc2bow(tokenize(para))
        print("输出测试文本可能的主题，每个主题包含6个词:")
        for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1 * tup[1]):
            print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 6)))

