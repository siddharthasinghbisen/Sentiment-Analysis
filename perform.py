import pandas as pd
import numpy as np
import webbrowser as w
import nltk
# import plotly
# import imp
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import matplotlib.patches as mpatches
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.utils.fixes import signature
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
# from tkinter import *
# from tkinter import ttk
# import sample2
# import nbcoderun
# import run
# # root = Tk()
# import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          filename=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(filename)


def split_into_lemmas(message):
    message = message.lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]


def percentage(part, whole):
    return 100 * float(part) / float(whole)

def init():
    prediction = dict()
    test_size = 0.2
    htmlpath = "D:\\xampp\\htdocs\\perform"
    file_path = "D:\\xampp\\htdocs\\perform\\15KTestData.txt"
    df = pd.read_csv(file_path, delimiter="\t", names=["sentiment", "text"], encoding='latin-1')
    df.text = df.text.astype(str)

    X_train, X_test, y_train, y_test = train_test_split(df.text, df.sentiment, test_size=test_size, shuffle=False)
    '''
    print("Xtrain: ", end =' ')
    print(X_train[0])
    print("X_test: ", end =' ')
    print(X_test[0])
    print("Ytest: ", end =' ')
    print(y_train[0])
    print("Y test: ", end =' ')
    print(y_test)
    '''

    # name = df.toxic.name
    # for i in X_train:
    #     print(i)

    # SGDClassifier
    text_clf_SGDClassifier = Pipeline([('vect', CountVectorizer(analyzer=split_into_lemmas, ngram_range=(2, 4),
                                                                stop_words='english', lowercase=True)),
                                       ('tfidf', TfidfTransformer()),
                                       ('clf', SGDClassifier()),
                                       ])
    text_clf_SGDClassifier.fit(X_train, y_train)
    y_scoreSGD = text_clf_SGDClassifier.decision_function(X_test)

    # LogisticRegression
    text_clf_LogisticRegression = Pipeline([('vect', CountVectorizer(analyzer=split_into_lemmas, ngram_range=(2, 4),
                                                                     stop_words='english', lowercase=True)),
                                            ('tfidf', TfidfTransformer()),
                                            ('clf', LogisticRegression()),
                                            ])
    text_clf_LogisticRegression.fit(X_train, y_train)
    y_scoreRegression = text_clf_LogisticRegression.decision_function(X_test)

    # Multinomial Naive Bayes
    text_clf_MultinomialNB = Pipeline([('vect', CountVectorizer(analyzer=split_into_lemmas, ngram_range=(2, 4),
                                                                stop_words='english', lowercase=True)),
                                       ('tfidf', TfidfTransformer()),
                                       ('clf', MultinomialNB()),
                                       ])
    text_clf_MultinomialNB.fit(X_train, y_train)
    # y_scoreNB = text_clf_MultinomialNB.decision_function(X_test)

    # Support vector machine
    text_clf_SVC = Pipeline([('vect',
                              CountVectorizer(analyzer=split_into_lemmas, ngram_range=(2, 4), stop_words='english',
                                              lowercase=True)),
                             ('tfidf', TfidfTransformer()),
                             ('clf', SVC(kernel='linear')),
                             ])
    text_clf_SVC.fit(X_train, y_train)
    y_scoreSVC = text_clf_SVC.decision_function(X_test)

    predicted_SVC = text_clf_SVC.predict(X_test)
    prediction['SVC'] = predicted_SVC
    predicted_MultinomialNB = text_clf_MultinomialNB.predict(X_test)
    prediction['MultinomialNB'] = predicted_MultinomialNB
    predicted_LogisticRegression = text_clf_LogisticRegression.predict(X_test)
    prediction['LogisticRegression'] = predicted_LogisticRegression
    predicted_SGDClassifier = text_clf_SGDClassifier.predict(X_test)
    prediction['SGDClassifier'] = predicted_SGDClassifier

    print("----------------------------------------------------------------------------")
    print("SVC!")
    # recall_values['SVC'] = send_recall(classification_report(y_test, predicted_SVC))
    print (classification_report(y_test, predicted_SVC))
    accuracyScoreSVC = accuracy_score(y_test, predicted_SVC)
    accuracyPercentSVC = accuracyScoreSVC * 100
    print('SVC: Accuracy Percentage: %s' % accuracyPercentSVC)

    # precision1, recall1, _ = precision_recall_curve(y_test, y_scoreSVC)

    # Computer confusion matrix
    array1 = confusion_matrix(y_test, predicted_SVC)
    reviewsnum1 = array1[0][0] + array1[0][1] + array1[1][0] + array1[1][1]
    plot_confusion_matrix(array1,
                          normalize=False,
                          target_names=['0', '1'],
                          title="Support Vector Machine",
                          filename='SVC')
    # Pie chart
    labels = 'Positive', 'Negative'
    sizes = [percentage(array1[1][0] + array1[1][1], reviewsnum1), percentage(array1[0][0] + array1[0][1], reviewsnum1)]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.savefig('SVC-Pie')

    # Writing to HTML file
    with open('Support-Vector-Machine.html', 'w') as myFile:
        myFile.write('<html><head><link rel="stylesheet" href="styles.css"></head>')
        myFile.write('<title>Support Vector machine</title>')
        myFile.write('<body>')
        myFile.write('<div id="content"><p>Accuracy Percentage: %s</p></div>' % accuracyPercentSVC)

        myFile.write('<h5>Total number of reviews: %s </h5>' % reviewsnum1)
        if (array1[1][1] >= array1[0][0]):
            myFile.write(
                '<h5>Since there are more number of POSITIVE Reviews: <strong> %s </strong>, Hence the general sentiment is <strong>Positive!</strong></h5> Go for it!!' %
                array1[1][1])
        else:
            myFile.write(
                '<h5>Since there are more number of Negative Reviews: <strong> %s </strong>, Hence the general sentiment is <strong>Negative!</strong></h5> It is a bad purchase' %
                array1[0][0])
        myFile.write('<h3>Following is the Confusion Matrix and Pie chart.</h3>')
        myFile.write('<div class="row">')
        myFile.write('<div class="column">')
        myFile.write('<img src="SVC.png" alt="Snow" style="width:100%"> </div>')
        myFile.write('<div class="column">')
        myFile.write('<img src="SVC-Pie.png" alt="Snow" style="width:100%"> </div></div>')
        # myFile.write('<img src="graph.png" >')

        myFile.write('</body>')
        myFile.write('</html>')
        myFile.close()
    w.open(htmlpath + 'Support-Vector-Machine.html')
    '''
    print(
        "------------------------------------------------------------------------------------------------------------------------")
    print('\n Multinomial Naive Bayes')
    print (classification_report(y_test, predicted_MultinomialNB))
    # average_precisionNB = average_precision_score(y_test, y_scoreNB)
    accuracyScoreNB = accuracy_score(y_test, predicted_MultinomialNB)
    accuracyPercentNB = accuracyScoreNB * 100
    print('Multinomial Naive Bayes: Accuracy Percentage: %s' % accuracyPercentNB)

    # precision1, recall1, _ = precision_recall_curve(y_test, y_scoreSVC)

    # Computer confusion matrix
    array1 = confusion_matrix(y_test, predicted_MultinomialNB)
    reviewsnum2 = array1[0][0] + array1[0][1] + array1[1][0] + array1[1][1]
    plot_confusion_matrix(array1,
                          normalize=False,
                          target_names=['0', '1'],
                          title="Multinomial Naive Bayes",
                          filename='Multinomial_Naive_Bayes')
    # Pie chart
    labels = 'Positive', 'Negative'
    sizes = [percentage(array1[1][0] + array1[1][1], reviewsnum2), percentage(array1[0][0] + array1[0][1], reviewsnum2)]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.savefig('NB-Pie')

    # Writing to HTML file
    with open('Multinomial-Naive-Bayes.html', 'w') as myFile:
        myFile.write('<html><head><link rel="stylesheet" href="styles.css"></head>')
        myFile.write('<title>Multinomial Naive Bayes</title>')
        myFile.write('<body>')
        myFile.write('<div id="content"><p>Accuracy Percentage: %s</p></div>' % accuracyPercentNB)
        myFile.write('<h5>Total number of reviews: %s </h5>' % reviewsnum2)
        if (array1[1][1] >= array1[0][0]):
            myFile.write(
                '<h5>Since there are more number of POSITIVE Reviews: <strong> %s </strong>, Hence the general sentiment is <strong>Positive!</strong></h5>' %
                array1[1][1])
        else:
            myFile.write(
                '<h5>Since there are more number of NEGATIVE Reviews: <strong> %s </strong>, Hence the general sentiment is <strong>Negative!</strong></h5>' %
                array1[0][0])
        myFile.write('<h3>Following is the <strong>Confusion Matrix</strong> and <strong>Pie Chart</strong>.</h3>')
        myFile.write('<div class="row">')
        myFile.write('<div class="column">')
        myFile.write('<img src="Multinomial_Naive_Bayes.png" alt="Snow" style="width:100%"> </div>')
        myFile.write('<div class="column">')
        myFile.write('<img src="NB-Pie.png" alt="Snow" style="width:100%"> </div></div>')
        # myFile.write('<img src="graph.png" >')

        myFile.write('</body>')
        myFile.write('</html>')
        myFile.close()
    # print('Multinomial Naive Bayes: Average precision-recall score: {0:0.2f}'.format(average_precisionNB))
    w.open(htmlpath + 'Multinomial-Naive-Bayes.html')

    print(
        "----------------------------------------------------------------------------------------------------------------------")
    print("\n Logistic Regression")
    # recall_values['LR'] = send_recall(classification_report(y_test, predicted_LogisticRegression))
    print (classification_report(y_test, predicted_LogisticRegression))
    # average_precisionNB = average_precision_score(y_test, y_scoreNB)
    accuracyScoreLR = accuracy_score(y_test, predicted_LogisticRegression)
    accuracyPercentLR = accuracyScoreLR * 100
    print('Logistic Regression: Accuracy Percentage: %s' % accuracyPercentLR)

    # precision1, recall1, _ = precision_recall_curve(y_test, y_scoreSVC)

    # Computer confusion matrix
    array1 = confusion_matrix(y_test, predicted_LogisticRegression)
    reviewsnum3 = array1[0][0] + array1[0][1] + array1[1][0] + array1[1][1]
    plot_confusion_matrix(array1,
                          normalize=False,
                          target_names=['0', '1'],
                          title="Logistic Regression",
                          filename='Logistic_Regression')

    # Pie chart
    labels = 'Positive', 'Negative'
    sizes = [percentage(array1[1][0] + array1[1][1], reviewsnum3), percentage(array1[0][0] + array1[0][1], reviewsnum3)]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.savefig('LR-Pie')

    # Writing to HTML file
    with open('Logistic-Regression.html', 'w') as myFile:
        myFile.write('<html><head><link rel="stylesheet" href="styles.css"></head>')
        myFile.write('<title>Logistic Regression</title>')
        myFile.write('<body>')
        myFile.write('<div id="content"><p>Accuracy Percentage: %s</p></div>' % accuracyPercentLR)

        myFile.write('<h5>Total number of reviews: %s </h5>' % reviewsnum3)
        if (array1[1][1] >= array1[0][0]):
            myFile.write(
                '<h5>Since there are more number of POSITIVE Reviews: <strong> %s </strong>, Hence the general sentiment is <strong>Positive!</strong></h5>' %
                array1[1][1])
        else:
            myFile.write(
                '<h5>Since there are more number of NEGATIVE Reviews: <strong> %s </strong>, Hence the general sentiment is <strong>Negative!</strong></h5>' %
                array1[0][0])
        myFile.write('<h3>Following is the <strong>Confusion Matrix</strong> and <strong>Pie Chart</strong>.</h3>')
        myFile.write('<div class="row">')
        myFile.write('<div class="column">')
        myFile.write('<img src="Logistic_Regression.png" alt="Snow" style="width:100%"> </div>')
        myFile.write('<div class="column">')
        myFile.write('<img src="LR-Pie.png" alt="Snow" style="width:100%"> </div></div>')
        # myFile.write('<img src="graph.png" >')

        myFile.write('</body>')
        myFile.write('</html>')
        myFile.close()
    w.open(htmlpath + 'Logistic-Regression.html')

    print("----------------------------------------------------------------------------")
    print("Stochastic Gradient Descent")
    # recall_values['SGD'] = send_recall(classification_report(y_test, predicted_SGDClassifier))
    print (classification_report(y_test, predicted_SGDClassifier))
    accuracyScoreSGD = accuracy_score(y_test, predicted_SGDClassifier)
    accuracyPercentSGD = accuracyScoreSGD * 100
    print('Stochastic Gradient Descent: Accuracy Percentage: %s' % accuracyPercentSGD)

    array1 = confusion_matrix(y_test, predicted_SGDClassifier)
    reviewsnum4 = array1[0][0] + array1[0][1] + array1[1][0] + array1[1][1]

    plot_confusion_matrix(array1,
                          normalize=False,
                          target_names=['0', '1'],
                          title="Stochastic Gradient Descent",
                          filename='Stochastic_Gradient_Descent')

    # Pie chart
    labels = 'Positive', 'Negative'
    sizes = [percentage(array1[1][0] + array1[1][1], reviewsnum4), percentage(array1[0][0] + array1[0][1], reviewsnum4)]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.savefig('SGD-Pie')

    # Writing to HTML file
    with open('Stochastic-Gradient-Descent.html', 'w') as myFile:
        myFile.write('<html><head><link rel="stylesheet" href="styles.css"></head>')
        myFile.write('<title>Stochastic Gradient Descent</title>')
        myFile.write('<body>')
        myFile.write('<div id="content"><p>Accuracy Percentage: %s</p></div>' % accuracyPercentSGD)

        myFile.write('<h5>Total number of reviews: %s </h5>' % reviewsnum4)
        if (array1[1][1] >= array1[0][0]):
            myFile.write(
                '<h5>Since there are more number of POSITIVE Reviews: <strong> %s </strong>, Hence the general sentiment is <strong>Positive!</strong></h5>' %
                array1[1][1])
        else:
            myFile.write(
                '<h5>Since there are more number of NEGATIVE Reviews: <strong> %s </strong>, Hence the general sentiment is <strong>Negative!</strong></h5>' %
                array1[0][0])
        myFile.write('<h3>Following is the <strong>Confusion Matrix</strong> and <strong>Pie Chart</strong>.</h3>')
        myFile.write('<div class="row">')
        myFile.write('<div class="column">')
        myFile.write('<img src="Stochastic_Gradient_Descent.png" alt="Snow" style="width:100%"> </div>')
        myFile.write('<div class="column">')
        myFile.write('<img src="SGD-Pie.png" alt="Snow" style="width:100%"> </div></div>')
        # myFile.write('<img src="graph.png" >')

        myFile.write('</body>')
        myFile.write('</html>')
        myFile.close()

    w.open(htmlpath + 'Stochastic-Gradient-Descent.html')

    # average_precisionSGD = average_precision_score(y_test, y_scoreSGD)

    # print('SGD Classifier: Average precision-recall score: {0:0.2f}'.format(average_precisionSGD))
    # precision3, recall3, _ = precision_recall_curve(y_test, y_scoreSGD)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument

    # plt.step(recall3, precision3, color='b', alpha=0.2, where='post')
    # plt.plot(recall3, precision3, alpha=0.2, color='b')

    # blue_patch = mpatches.Patch(color='blue', label='SGD: '+str(average_precisionSGD))
    # plt.legend(handles=[green_patch, red_patch, blue_patch])

    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('2-class Precision-Recall curve: using SGD, SVM and Logistic Regression')
    # plt.show()
    # plt.savefig('graph')
    # plotly.offline.plot(plt,filename='plot.html')
    # plotly.offline.plot(fig, filename = 'filename.html', auto_open=False)

    print("----------------------------------------------------------------------------")
    # reviewsnum = recall_values['SVC']['positiveSupport'] + recall_values['SVC']['negativeSupport']

    # if(recall_values['SVC']['positiveSupport'] >= recall_values['SVC']['negativeSupport']):
    # labelvar.set("The Product has got many Positive reviews. Hence the General Sentiment is positive")
    # html_str = """
    #             <h4> Since there are more number of positive Values with  %04d , following are positve recalls of different classification methods</h4>
    #             <table border=1>
    #                  <tr>
    #                    <th>Recall</th>
    #                    <th>SVC</th>
    #                    <th>SGD</th>
    #                    <th>Logistic</th>
    #                  </tr>
    #                  <indent>
    #                  <% for i in recall_values: }
    #                    <tr>
    #                    <td></td>
    #                      <td>{ i['SVC']['positiveRecall'] }</td>
    #                      <td>{ i['SGD']['positiveRecall'] }</td>
    #                      <td>{ i['LR']['positiveRecall'] }</td>

    #                    </tr>
    #                  </indent>
    #             </table>
    #             <br />
    #             <img src="graph.png" >
    #             """

    #     with open('positive.html', 'w') as myFile:
    #         myFile.write('<html><body>')
    #         myFile.write('<h5>Total number of reviews: %s </h5>' % reviewsnum)
    #         myFile.write('<h3>Following are Recalls of Good reviews detected.</h3>')

    #         myFile.write('<table>')

    #         myFile.write('<tr><th>Recall</th><th>SVC</th><th>SGD</th><th>Logistic</th></tr>')
    #         myFile.write('<tr><td></td>')
    #         myFile.write('<td>%s</td>' % recall_values['SVC']['positiveRecall'])

    #         myFile.write('<td>%s</td>' % recall_values['SGD']['positiveRecall'])

    #         myFile.write('<td>%s</td></tr>' % recall_values['LR']['positiveRecall'])

    #         myFile.write('</tr>')
    #         myFile.write('</table> <br />')

    #         myFile.write('<img src="graph.png" >')
    #         # myFile.write('<img src="graph.png" >')

    #         myFile.write('<h5> Since there are more number of POSITIVE Reviews: <strong> %s </strong>, Hence the general sentiment is <strong>Positive!</strong></h5>' % recall_values['SVC']['positiveSupport'])

    #         myFile.write('</body>')
    #         myFile.write('</html>')
    #     # Html_file= open("positive.html","w")
    #     # Html_file.write(html_str)
    #     # Html_file.close()
    #     w.open('C:\\Users\\vishn\\Desktop\\sentiment_analysis\\Defense\\codes\\sample\\postive.html')
    #     # return True
    # else:
    #     # labelvar.set("The Product has got many Negative reviews. Hence the General Sentiment is negative")

    #     with open('negative.html', 'w') as myFile:
    #         myFile.write('<html><body>')
    #         myFile.write('<h5>Total number of reviews: %s</h5>' % reviewsnum)
    #         myFile.write('<h3>Following are Recalls of Bad Reviews detected</h3>')
    #         myFile.write('<table>')
    #         myFile.write('<tr><th>Recall</th><th>SVC</th><th>SGD</th><th>Logistic</th></tr>')
    #         myFile.write('<tr><td></td>')
    #         myFile.write('<td>%s</td>' % recall_values['SVC']['negativeRecall'])

    #         myFile.write('<td>%s</td>' % recall_values['SGD']['negativeRecall'])

    #         myFile.write('<td>%s</td></tr>' % recall_values['LR']['negativeRecall'])

    #         myFile.write('</tr>')
    #         myFile.write('</table><br />')
    #         myFile.write('<img src="graph.png" >')
    #         myFile.write('<h5> Since there are more number of Negative Reviews: <strong> %s </strong>, Hence the general sentiment is <strong>Negative!</strong> </h5>' % recall_values['SVC']['negativeSupport'])

    #         myFile.write('</body>')
    #         myFile.write('</html>')
    #     # Html_file= open("positive.html","w")
    #     # Html_file.write(html_str)
    #     # Html_file.close()
    #     w.open('C:\\Users\\vishn\\Desktop\\sentiment_analysis\\Defense\\codes\\sample\\negative.html')

   
    '''


init()
