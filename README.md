# nlp_topic_extraction
lda 文本主题提取
使用 lda 主题模型

—————————————————————————————————————————————————

本项目开发环境为：Windows11+PyCahrm2022+Anaconda3（Python 3.9）

使用 gensim、nltk、matplotlib、docx 等第三方库。

—————————————————————————————————————————————————

主要技术路线：

1.使用 docx 库读取文本。

2.使用 gensim 库对文本进行分词。

3.使用 nltk 库的 wordnet 和 omw-1.4 词表对训练文本进行词形还原；使用 gensim 库的 stopwords 词表对训练文本去停用词 。

4.从 1~100 中选择合适的主题数训练 lda 模型。

5.根据上一步选择的主题数训练 lda 模型。

6.输出测试文本的可能的主题。

—————————————————————————————————————————————————

项目架构：
source/data：存放训练文本、测试文本。
source/dictionary_and_corpus：保存分词生成的字典（dictionary）和语料库（corpus）。（该部分功能在注释内容中）
source/model：保存训练好的lda模型供下次使用。（该部分功能在注释内容中）

main.py：主文件。
lda.py：实现分词、生成语料库、输出语料库等功能。
read_articles.py：实现读取docx文件功能。
