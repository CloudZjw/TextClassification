from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import sklearn


# 停用词列表
def stopwords_list(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def read_words(filepath):
    words = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return words


def read_tag_list(filepath):
    tags = [int(line.strip()) for line in open(filepath, 'r').readlines()]
    return tags


# def get_feature(stpwrdpth, convspth):
#     stpwrdlst = stopwords_list(stpwrdpth)
#     convs = convs_list(convspth)
#
#     vectorizer = TfidfVectorizer(stop_words=stpwrdlst, token_pattern='\\b\\w+\\b')
#     vectorizer.fit(convs)
#     # 得到词袋信息
#     # bag_of_words = vectorizer.get_feature_names()
#     # print("Bag of words: ")
#     # print(bag_of_words)
#
#     # 将文本内容转化为词袋向量
#     words_vec = vectorizer.fit_transform(convs)
#
#     # print("Vectorized convs: ")
#     # print(X_vec.toarray())
#
#     return words_vec


def model_train_bayes(x_train, y_train, x_val, y_val, stopwords_):
    # TF-IDF向量化
    # u'(?u)\w+' '\\b\\w+\\b'
    vectorizer = TfidfVectorizer(stop_words=stopwords_, token_pattern='\\b\\w+\\b')
    x_train = vectorizer.fit_transform(x_train)
    x_val = vectorizer.transform(x_val)

    # 使用朴素贝叶斯多项式模型
    # 平滑系数 = 1
    mlb = MultinomialNB(alpha=1)
    mlb.fit(x_train, y_train)

    # # 训练集的训练结果
    # accuracy, auc = evaluate(mlb, x_train, y_train)
    # print("训练集正确率：%.4f%%\n" % (accuracy * 100))
    # print("训练AUC值： %.6f\n" % (auc))
    #
    # # 测试集上的评测结果
    # accuracy, auc = evaluate(mlb, x_val, y_val)
    # print("测试集正确率：%.4f%%\n" % (accuracy * 100))
    # print("测试AUC值：%.6f\n" % (auc))

    y_predict = mlb.predict(x_val)
    cfm = metrics.confusion_matrix(y_val, y_predict)
    label = list(set(y_predict))
    print("微平均准确率：" + str(metrics.precision_score(y_val, y_predict, label, average='micro')))
    print("宏平均准确率：" + str(metrics.precision_score(y_val, y_predict, label, average='macro')))
    print("召回率：" + str(metrics.recall_score(y_val, y_predict, average='micro')))
    print("F1 score: " + str(metrics.f1_score(y_val, y_predict, average='weighted')))
    print(metrics.classification_report(y_val, y_predict))

    return mlb, vectorizer


def model_train_svm(x_train, y_train, x_val, y_val, stopwords_):
    # TF-IDF向量化
    # u'(?u)\w+' '\\b\\w+\\b'
    vectorizer = TfidfVectorizer(stop_words=stopwords_, token_pattern='\\b\\w+\\b')
    x_train = vectorizer.fit_transform(x_train)
    x_val = vectorizer.transform(x_val)

    # 使用朴素贝叶斯多项式模型
    # 平滑系数 = 1
    mlb = SVC(kernel='linear')
    mlb.fit(x_train, y_train)

    # # 训练集的训练结果
    # accuracy, auc = evaluate(mlb, x_train, y_train)
    # print("训练集正确率：%.4f%%\n" % (accuracy * 100))
    # print("训练AUC值： %.6f\n" % (auc))
    #
    # # 测试集上的评测结果
    # accuracy, auc = evaluate(mlb, x_val, y_val)
    # print("测试集正确率：%.4f%%\n" % (accuracy * 100))
    # print("测试AUC值：%.6f\n" % (auc))

    y_predict = mlb.predict(x_val)
    cfm = metrics.confusion_matrix(y_val, y_predict)
    label = list(set(y_predict))
    print("微平均准确率：" + str(metrics.precision_score(y_val, y_predict, label, average='micro')))
    print("宏平均准确率：" + str(metrics.precision_score(y_val, y_predict, label, average='macro')))
    print("召回率：" + str(metrics.recall_score(y_val, y_predict, average='micro')))
    print("F1 score: " + str(metrics.f1_score(y_val, y_predict, average='weighted')))
    print(metrics.classification_report(y_val, y_predict))

    return mlb, vectorizer


def evaluate(model, x, y):
    accuracy = model.score(x, y)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, model.predict_proba)
    return accuracy, sklearn.metrics.auc(fpr, tpr)


if __name__ == "__main__":
    # 加载数据
    stopwords = stopwords_list('resources//cn_stopwords.txt')
    words_train = read_words('resources//words_train.txt')
    words_val = read_words('resources//words_val.txt')
    tag_list_train = read_tag_list('resources//tags_train.txt')
    tag_list_val = read_tag_list('resources//tags_val.txt')
    # print(tag_list_train)
    mlb, vectorizer = model_train_svm(words_train, tag_list_train, words_val, tag_list_val, stopwords)
