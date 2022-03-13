from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from textblob import TextBlob
from googletrans import Translator
import pandas as pd
import urllib.request
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from flask import Flask, app, render_template, request, flash
import tweepy
import csv
import nltk
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
nltk.download('punkt')
nltk.download('stopwords')


app = Flask(__name__, static_folder="static/assets")
app.config['SECRET_KEY'] = 'laila'

scrapping_result = []


def scrapping_data(query, jumlah):
    api_key = "XaYBZDIN7j6xVHdPRIfOsu6mJ"
    api_secret_key = "CQKtU7Bpi8xmzhfLgqFguABTBOvBUwS8KQMdQk5A1HAttLKLSx"
    access_token = "1280139539361566721-e7s1tA20RyOTUAqAPT5dxiMawuUOWe"
    access_token_secret = "7PhFWlLR6eHVNbmtLqKjBaGJnQjQRUAp9fkcL2CUpwb4B"

    auth = tweepy.OAuthHandler(api_key, api_secret_key)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    filter = " -filter:retweets"
    # Membuat File CSV
    # membuat file scrapping csv
    file = open('static/assets/files/Data Hasil Scrapping.csv',
                'w', newline='', encoding='utf-8')
    writer = csv.writer(file)

    scrapping_result.clear()

    for tweet in tweepy.Cursor(api.search, q=query + filter, lang='id', tweet_mode="extended").items(int(jumlah)):

        # Menuliskan data ke csv
        tweets = [tweet.created_at, tweet.user.screen_name,
                  tweet.full_text.replace('\n', '')]

        scrapping_result.append(tweets)

        writer.writerow(tweets)


preprocessing_result = []


def prepropecossing_twitter():
    # Membuat File CSV
    file = open('static/assets/files/Data Hasil Preprocessing.csv',
                'w', newline='', encoding='utf-8')
    writer = csv.writer(file)

    preprocessing_result.clear()

    with open("static/assets/files/Data Hasil Scrapping.csv", "r", encoding='utf-8') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        preprocessing_result.clear()
        for row in readCSV:
            # proses clean
            clean = ' '.join(
                re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", row[2]).split())
            clean = re.sub("\d+", "", clean)

            # proses casefold
            casefold = clean.casefold()

            # proses tokenize
            tokenizing = nltk.tokenize.word_tokenize(casefold)

            # proses stop removal
            # mengambil data stop word dari library
            stop_factory = StopWordRemoverFactory().get_stop_words()
            # menambah stopword sendiri
            more_stop_word = ["apa", "yg"]
            # menggabungkan stopword library + milik sendiri
            data = stop_factory + more_stop_word

            dictionary = ArrayDictionary(data)
            str = StopWordRemover(dictionary)
            stop_wr = nltk.tokenize.word_tokenize(str.remove(casefold))

            # proses stemming
            kalimat = ' '.join(stop_wr)
            factory = StemmerFactory()
            # mamanggil fungsi stemming
            stemmer = factory.create_stemmer()
            stemming = stemmer.stem(kalimat)

            tweets = [row[0], row[1], row[2], clean,
                      casefold, tokenizing, stop_wr, stemming]
            preprocessing_result.append(tweets)

            writer.writerow(tweets)

    flash('Preprocessing Berhasil', 'preprocessing_category')


hasil_labelling = []


def labelling_process():
    # Membuat File CSV
    file = open('static/assets/files/Data Hasil Labelling.csv',
                'w', newline='', encoding='utf-8')
    writer = csv.writer(file)
    translator = Translator()

    with open("static/assets/files/Data Hasil Preprocessing.csv", "r", encoding='utf-8') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        hasil_labelling.clear()
        for row in readCSV:
            tweet = {}
            try:
                value = translator.translate(row[6], dest='en')
            except:
                print("Terjadi kesalahan", flush=True)

            terjemahan = value.text
            data_label = TextBlob(terjemahan)

            if data_label.sentiment.polarity > 0.0:
                tweet['sentiment'] = "Positif"
            elif data_label.sentiment.polarity == 0.0:
                tweet['sentiment'] = "Netral"
            else:
                tweet['sentiment'] = "Negatif"

            labelling = tweet['sentiment']
            tweets = [row[2], row[7], labelling]
            hasil_labelling.append(tweets)

            writer.writerow(tweets)
    flash('Labelling Berhasil', 'labelling_category')


df = None
df2 = None


akurasi = 0


def klasifikasi_data():
    global df
    global df2
    global akurasi
    # membca csv
    tweet = []
    y = []

    with open("static/assets/files/Data Hasil Labelling.csv", encoding='utf-8') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            tweet.append(row[1])
            y.append(row[2])

    vectorizer = TfidfVectorizer()
    vectorizer.fit(tweet)
    # tfidf = vectorizer.fit_transform(X_train)
    x = vectorizer.transform(tweet)

    # split data training dan testing
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=45)

    # naive bayes
    # naove bayes
    clf = MultinomialNB()
    clf.fit(x_train, y_train)

    predict = clf.predict(x_test)
    report = classification_report(y_test, predict, output_dict=True)
    # simpan ke csv
    clsf_report = pd.DataFrame(report).transpose()
    clsf_report.to_csv(
        'static/assets/files/Data Hasil Klasifikasi.csv', index=True)

    pickle.dump(vectorizer, open('static/assets/files/vec.pkl', 'wb'))
    pickle.dump(x, open('static/assets/files/tfidf.pkl', 'wb'))
    pickle.dump(clf, open('static/assets/files/model.pkl', 'wb'))

    unique_label = np.unique([y_test, predict])
    cmtx = pd.DataFrame(
        confusion_matrix(y_test, predict, labels=unique_label),
        index=['true:{:}'.format(x) for x in unique_label],
        columns=['pred:{:}'.format(x) for x in unique_label]
    )

    cmtx.to_csv(
        'static/assets/files/Data Hasil Confusion Matrix.csv', index=True)

    df = pd.read_csv(
        'static/assets/files/Data Hasil Confusion Matrix.csv', sep=",")
    df.rename(columns={'Unnamed: 0': ''}, inplace=True)

    df2 = pd.read_csv(
        'static/assets/files/Data Hasil Klasifikasi.csv', sep=",")
    df2.rename(columns={'Unnamed: 0': ''}, inplace=True)

    akurasi = round(accuracy_score(y_test, predict) * 100, 2)

    kalimat = ""

    for i in tweet:
        s = ("".join(i))
        kalimat += s

    urllib.request.urlretrieve(
        "https://firebasestorage.googleapis.com/v0/b/sentimen-97d49.appspot.com/o/Circle-icon.png?alt=media&token=b9647ca7-dfdb-46cd-80a9-cfcaa45a1ee4", 'love.png')
    mask = np.array(Image.open("love.png"))
    wordcloud = WordCloud(width=1600, height=800,
                          max_font_size=200, background_color='white', mask=mask)
    wordcloud.generate(kalimat)
    plt.figure(figsize=(12, 10))

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.savefig('static/assets/files/wordcloud.png')

    # diagram

    counter = dict((i, y.count(i)) for i in y)
    isPositive = 'Positif' in counter.keys()
    isNegative = 'Negatif' in counter.keys()
    isNeutral = 'Netral' in counter.keys()

    positif = counter["Positif"] if isPositive == True else 0
    negatif = counter["Negatif"] if isNegative == True else 0
    netral = counter["Netral"] if isNeutral == True else 0

    sizes = [positif, netral, negatif]
    labels = ['Positif', 'Netral', 'Negatif']
    plt.pie(sizes, labels=labels, autopct='%1.0f%%',
            shadow=True, textprops={'fontsize': 20})
    plt.savefig('static/assets/files/pie-diagram.png')

    # diagram batang
    # creating the bar plot

    plt.figure()

    plt.hist(y)

    plt.xlabel("Tweet tentang SpiderMan")
    plt.ylabel("Jumlah Tweet")
    plt.title("Presentase Sentimen Tweet Spiderman")
    plt.savefig('static/assets/files/bar-diagram.png')
    flash('Klasifikasi Berhasil', 'classification_category')


hasil_model_predict = []


def model_predict():
    global df
    global df2
    global akurasi
    # membca csv
    data = pd.read_csv(
        "static/assets/files/Data Hasil Labelling Model Predict.csv")
    tweet = data.iloc[:, 1]
    y = data.iloc[:, 2]

    # Vectorize text reviews to numbers
    # tfidf = joblib.load('templates/assets/files/tfidf.pkl')
    # nb = joblib.load('templates/assets/files/model.pkl')
    # vec = joblib.load('templates/assets/files/countvec.pkl')
    with open('templates/assets/files/model.pkl', 'rb') as handle:
        model = pickle.load(handle)

    with open('templates/assets/files/countvec.pkl', 'rb') as h:
        vec = pickle.load(h)

    with open('templates/assets/files/tfidf.pkl', 'rb') as t:
        tfidf = pickle.load(t)

    file = open('templates/assets/files/Data Hasil Model Predict.csv',
                'w', newline='', encoding='utf-8')
    writer = csv.writer(file)
    for i, line in data.iterrows():
        isi = line[1]
        label = line[2]
        # # transform cvector & tfidf
        transform_cvec = vec.transform([isi])
        transform_tfidf = tfidf.transform(transform_cvec)
        print(transform_tfidf)
        # predict start
        predic_result = model.predict(transform_tfidf)
        print(predic_result)

        data = [isi, predic_result[0], label]
        hasil_model_predict.append(data)
        writer.writerow(data)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/scrapping', methods=['GET', 'POST'])
def scrapping():
    if request.method == 'POST':
        query = request.form.get('query')
        jumlah = request.form.get('jumlah')
        if request.form.get('scrapping') == 'Scrapping':
            scrapping_data(query, jumlah)
            return render_template('scrapping.html', value=scrapping_result)
    return render_template('scrapping.html', value=scrapping_result)


ALLOWED_EXTENSION = set(['csv'])


def allowed_files(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION


@app.route('/preprocessing', methods=['GET', 'POST'])
def preprocessing():
    if request.method == 'POST':
        if request.form.get('upload') == 'upload':
            file = request.files['file']
            if not allowed_files(file.filename):
                flash('Format Salah', 'upload_category')
                return render_template('preprocessing.html', value=preprocessing_result)
            if file and allowed_files(file.filename):
                flash('Upload Berhasil', 'upload_category')
                file.save("static/assets/files/Data Hasil Scrapping.csv")
        if request.form.get('preprocessing') == 'Preprocessing':
            prepropecossing_twitter()
            return render_template('preprocessing.html', value=preprocessing_result)

    return render_template('preprocessing.html', value=preprocessing_result)


@app.route('/labelling', methods=['POST', 'GET'])
def labelling():
    if request.method == 'POST':
        if request.form.get('upload') == 'upload':
            file = request.files['file']
            if not allowed_files(file.filename):
                flash('Format Salah', 'upload_category')
                return render_template('labelling.html', value=hasil_labelling)
            if file and allowed_files(file.filename):
                flash('Upload Berhasil', 'upload_category')
                file.save("static/assets/files/Data Hasil Preprocessing.csv")

        hasil_labelling.clear()
        if request.form.get('labelling') == 'Labelling':
            labelling_process()

            return render_template('labelling.html', value=hasil_labelling)

    return render_template('labelling.html', value=hasil_labelling)


@app.route('/prediksimodel')
def prediksimodel():
    return render_template('prediksimodel.html')


@app.route('/visualisasi')
def visualisasi():
    return render_template('visualisasi.html')


@app.route('/klasifikasi',  methods=['POST', 'GET'])
def klasifikasi():
    if request.method == 'POST':
        if request.form.get('upload') == 'upload':
            file = request.files['file']
            if not allowed_files(file.filename):
                flash('Format Salah', 'upload_category')
                return render_template('klasifikasi.html')
            if file and allowed_files(file.filename):
                flash('Upload Berhasil', 'upload_category')
                file.save("static/assets/files/Data Hasil Labelling.csv")
        if request.form.get('klasifikasi') == 'Klasifikasi':
            klasifikasi_data()
            return render_template('klasifikasi.html', accuracy=akurasi, tables=[df.to_html(classes='table table-striped', index=False, justify='left')], titles=df.columns.values, tables2=[df2.to_html(classes='table table-striped', index=False, justify='left')], titles2=df2.columns.values)
    if akurasi == 0:
        return render_template('klasifikasi.html')
    else:
        return render_template('klasifikasi.html', accuracy=akurasi, tables=[df.to_html(classes='table table-striped', index=False, justify='left')], titles=df.columns.values, tables2=[df2.to_html(classes='table table-striped', index=False, justify='left')], titles2=df2.columns.values)


if __name__ == '__main__':
    app.run(debug=True)
