from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


def wordCloudData():
    arquivo = open('teste.txt', 'r').read()

    # create the word cloud
    wordcloud = WordCloud(background_color='white',
                          width=1024, height=768,
                          max_words=100).generate(arquivo)

    print('Word cloud created!')

    # display the cloud
    fig = plt.figure()
    fig.set_figwidth(14)
    fig.set_figheight(18)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    wordCloudData()
