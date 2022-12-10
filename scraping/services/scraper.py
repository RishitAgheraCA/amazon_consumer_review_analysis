import asyncio, re, pickle
import base64
import io
import urllib

from pyppeteer import launch
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

review_texts = []
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english'))
stopword_list = stopwords.words('english')

l = ["n'", 'nor', 'no', 'not']

suitable_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', """you're""", "you've",
                      "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                      'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',
                      'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these',
                      'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
                      'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                      'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                      'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
                      'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                      'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
                      'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
                      'will', 'just', 'don', "don't", 'should', 'play', 'playing', 'tablet', 'use', 'get', 'kindel',
                      'read', 'book', 'device', 'year', 'time', 'want', "should've", 'now', 'd', 'll', 'm', 'o', 're',
                      've', 'y']
replace_list = ['ain', 'hate', 'bad', 'worse', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn',
                'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', "n't", "n'"]

for i in stopword_list:
    if not any(words in i for words in l):
        suitable_stopwords.append(i)


# print(stopword_list)
# print(len(suitable_stopwords))


def cleanstr(text):
    text = text.lower()
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(' \d+', ' ', text)
    return text


def remove_stopwords(stmt):
    filtered_sentence = []
    stmt = stmt.lower()
    words = word_tokenize(stmt)

    for w in words:
        if w not in suitable_stopwords:
            if w not in replace_list:
                filtered_sentence.append(w)
            else:
                filtered_sentence.append(w + ' not')
    return " ".join(filtered_sentence)


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize(text):
    wl = WordNetLemmatizer()
    lemmatized_sentence = []
    # Tokenize the sentence
    words = word_tokenize(text)
    word_pos_tags = nltk.pos_tag(words)
    for idx, tag in enumerate(word_pos_tags):
        lemmatized_sentence.append(wl.lemmatize(tag[0], get_wordnet_pos(tag[1])))

    return " ".join(lemmatized_sentence)


async def scraper(url):
    browser = await launch({'headless': False})
    page = await browser.newPage()
    await page.goto(url)
    main_page = await page.waitForXPath('//*[@id="cm-cr-dp-review-list"]')
    page_html_text = await page.evaluate("""(element) => element.innerHTML""", main_page)
    review_ids = [i.split()[0] for i in re.findall(r'customer_review-(.*)', page_html_text)]
    review_ids = [i.replace('"', "") for i in review_ids]
    print(review_ids)
    for id in review_ids:
        comment_element = await page.waitForXPath(f'//*[@id="customer_review-{id}"]/div[4]/span/div/div[1]/span/text()')
        comment = await page.evaluate("""(element) => element.textContent""", comment_element)
        review_texts.append(comment)
    # print(review_texts)
    await browser.close()
    return review_texts


# asyncio.get_event_loop().run_until_complete(scraper())

def inference(reviews):
    prediction_map = {0: "Negative",
                      1: "Positive"}
    sentiments = []
    model = get_model()
    tfidf = get_vectorizer()
    pos = ""
    neg = ""
    for review in reviews:
        # print(review)
        text = cleanstr(review)
        # print('text clean:', text)
        text = remove_stopwords(text)
        # print('text stop:', text)
        text = (lemmatize(text))
        vec = tfidf.transform([text])
        # print('text:', text)
        # print('VEC:', vec)
        pre = model.predict(vec)
        # print('prediction:', pre[0])
        if pre == 1:
            pos += text
        else:
            neg += text
        sentiments.append(prediction_map.get(pre[0], 1))
    # print(sentiments)

    texts = {"positive": pos,
             "negative": neg}
    return sentiments, texts


def get_model():
    pickled_model = pickle.load(open('scraping\services\model\sentiment_classifier_model.pkl', 'rb'))
    return pickled_model


def get_vectorizer():
    vectorizer = pickle.load(open('scraping\services\model\_vectorizer.pk', 'rb'))
    return vectorizer

# inference(review_texts.copy())
def get_wordcloud(data):
    wordcloud = WordCloud().generate(data)
    wordcloud.generate(str(data))
    image = io.BytesIO()
    plt.savefig(image, format="png")
    image.seek(0)
    string = base64.b64encode(image.read())
    image_64 = "data:image/png;base64," + urllib.parse.quote_plus(string)
    return image_64