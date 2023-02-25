from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora
from gensim import models
from gensim.models.ldamodel import LdaModel
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import matplotlib.pyplot as plt
import matplotlib
import warnings
import nltk

# 不显示警告
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

# 第一次加载nltk时需要
print("正在下载 wordnet 语料库...")
nltk.download('wordnet')
print("正在下载 omw-1.4 语料库...")
nltk.download('omw-1.4')


def press_anything_continue():
    """
    输入任意键继续，输入exit结束程序
    """
    print("\n（输入任意键继续,输入 exit 结束...）\n")
    a = input()
    if a == 'exit':
        exit()


def reducing_words(text):
    """
    词形还原：将单词简化为词根或词干称为词形还原。
    :param text: 需要处理的词
    :return: 词性还原后的词
    """
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def tokenize(text):
    """
    分词、去停用词。STOPWORDS -是指Stone, Denis, Kwantes(2010)的stopwords集合
    :param text: 需要处理的文本
    :return: 分词、去停用词后的文本
    """
    return [reducing_words(token) for token in simple_preprocess(text) if token not in STOPWORDS]


def split_and_process(fulltext):
    """
    文本处理: 分词、生成字典和词袋
    :param fulltext: 训练文本
    :return: dictionary 字典, corpus 词袋, corpus_tfidf TF-TDF 处理后的词袋
    """
    # 分词 去停用词 对列表fulltext中所有段落进行分词和去停用词处理,得到对应的“词”序列集。
    processed_docs = [tokenize(doc) for doc in fulltext]
    # 建立字典 为每个出现在语料库中的单词分配了一个独一无二的整数编号
    dictionary = corpora.Dictionary(processed_docs)
    """
    # 保存为dictionary_and_corpus/dictionary.dict
    dictionary.save('dictionary_and_corpus/dictionary.dict')
    # 选择词频率大于10次，却又不超过文档大小的20%的词(注：由于规模缩小，所以有些词的id可能会改变)
    dictionary.filter_extremes(no_below=20, no_above=0.1)
    """
    # 建立词袋
    corpus = [dictionary.doc2bow(pdoc) for pdoc in processed_docs]
    """
    # 保存为dictionary_and_corpus/corpus.mm
    corpora.MmCorpus.serialize('dictionary_and_corpus/corpus.mm', corpus)
    """
    # TF-IDF 是一种通过计算词的权重来衡量文档中每个词的重要性的技术。在 TF-IDF 向量中，每个词的权重与该词在该文档中的出现频率成反比。
    corpus_tfidf = models.TfidfModel(corpus)[corpus]
    return dictionary, corpus, corpus_tfidf


def show_dictionary_and_corpus(dictionary, corpus, corpus_tfidf):
    """
    输出文本处理后的训练语料（字典、词袋）
    :param dictionary: 字典
    :param corpus: 词袋
    :param corpus_tfidf: 使用TF-IDF算法处理后的词袋
    """
    press_anything_continue()
    # 输出字典
    print("输出文本处理后的训练语料（字典、词袋）：")
    print("字典dictionary：\n", dictionary)
    print("字典中共有：\n", len(dictionary), "条不同的“词”")
    # 输出字典中每个词及对应编号
    print("字典中每个词及其对应编号：\n", dictionary.token2id)
    press_anything_continue()
    # 输出词袋
    print("词袋corpus:\n")
    print(corpus)
    press_anything_continue()
    # 输出TF-IDF算法处理后的词袋
    print("TF-IDF算法处理后的词袋corpus_tfidf：\n")
    for doc in corpus_tfidf:
        print(doc)


def find_optimal_topic_num(num, dictionary, corpus_tfidf):
    """
    :param num: 最大主题数
    :param dictionary: 字典
    :param corpus_tfidf: 使用TF-IDF算法处理后的词袋
    :return: 返回perplexity最小时的主题数
    """
    press_anything_continue()
    print("在 1~{} 之间选择perplexity最小的主题数:\n".format(num))
    print("（这里需要等待一段时间...）\n")
    x = range(1, num + 1)
    y = [perplexity(i, dictionary, corpus_tfidf) for i in x]
    # 数据可视化
    plt.plot(x, y)
    plt.xlabel('主题数目')
    plt.ylabel('perplexity')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.title('主题数-perplexity变化情况')
    plt.show()
    # 返回perplexity最小时的主题数
    return y.index(min(y)) + 1


# 计算困惑度
def perplexity(num_topics, dictionary, corpus_tfidf):
    """
    :param num_topics: lda 模型的主题数目
    :param dictionary: 字典
    :param corpus_tfidf: 使用TF-IDF算法处理后的词袋
    :return: 返回主题数为 num_topics 时的困惑度 perplexity
    """
    lda_model = LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary, passes=5)
    # 输出主题数 num_topics 和困惑度 perplexity
    print("num_topics: {}\t perplexity: {}".format(num_topics, lda_model.log_perplexity(corpus_tfidf)))
    return lda_model.log_perplexity(corpus_tfidf)

