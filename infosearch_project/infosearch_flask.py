# для препроцессинга
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize
import pymorphy2

# для анализа данных и реализации поиска
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

# для веб-приложения и интерфейса
from flask import Flask
from flask import url_for, render_template, request, redirect


### ### ###
### ПОИСКОВОЙ МЕХАНИЗМ ###
### ### ###

# Функция для получения текста вопросов.
def getting_texts():
    # Для ускоренной настройки модели считывается только 1000 первых строчек, но можно указать 'None' для чтения всего файла.
    nRowsRead = 10001
    data = pd.read_csv("E:\\infosearch_project\\quora_question_pairs_rus.csv", delimiter=',', header = 0, nrows = nRowsRead).dropna()
    all_questions_dataset = pd.concat([data['question1'], data['question2']], ignore_index=True)
    all_questions_list = all_questions_dataset.tolist()
    return all_questions_list


# Настраиваем лемматизатор, устанавливаем стоп-слова и немного расширяем пунктуацию.
morph_analyze = pymorphy2.MorphAnalyzer()
russian_stopwords = stopwords.words("russian")
punctuation += '«»…—'


# Функция для препроцессинга текста
def preprocessing_text(text):
    lemmas = []
    # Переводим всё в нижний регистр шрифта и делим по пробелам
    pre_tokens = word_tokenize(text.lower())
    # Избавляемся от стоп-слов и пунктуации, лемматизируем
    for one_pre_token in pre_tokens:
        if one_pre_token not in punctuation and one_pre_token not in russian_stopwords:
            one_lemma = morph_analyze.normal_forms(one_pre_token)[0]
            lemmas.append(one_lemma)
    # Превращаем в строки с леммами, разделёнными пробелами
    processed_text = ' '.join(lemmas)
    return processed_text


# Функция, которая создаёт последовательности: исходные вопросы из датасета, они же после препроцессинга, их id.
def sequencing_texts(texts_list):
    # Вспомогательная переменная, здесь будут храниться все тексты, прошедшие предпроцессинг.
    all_questions_processed = []
    # А здесь будут храниться все тексты, которые не прошли предпроцессинг.
    all_questions_raw = texts_list
    for one_question in all_questions_raw:
        # Лемматизируем текст вопроса, избавляемся от стоп-слов и ненужных символов.
        processed_question = preprocessing_text(one_question)
        # Записываем прошедший предобработку вопрос в переменную.
        all_questions_processed.append(processed_question)
    # Делаем массив с id вопросов.
    all_question_ids = [i for i in range(1, len(all_questions_processed) + 1)]
    return all_questions_raw, all_questions_processed, all_question_ids


# Функция, которая подготавливает поисковой запрос к обработке.
def query_preprocessing(query):
    # Разбиваем запрос на слова.
    processed_query = preprocessing_text(query)
    query_list = processed_query.split()
    # Применяем уже подготовленный векторайзер.
    query_matrix = tfidf_vectorizer.transform(query_list)
    return query_matrix


# Функция поиска: находит самые подходящие данные через косинусное расстояние.
def searching_cosine_similarity(query_matrix):
    cosine_similarities = linear_kernel(query_matrix[0:1], tfidf).flatten()
    # Находим лучшие мэтчи и их метрики.
    related_questions_ids = cosine_similarities.argsort()[:-6:-1]
    related_questions_metrics = cosine_similarities[related_questions_ids]
    res_number = 1
    # Готовим вывод для функции
    search_results = []
    for i in related_questions_ids:
        metr = related_questions_metrics[res_number-1]
        new_result = '№ ' + str(res_number) + ', метрика = ' + str(metr) + ', вопрос: ' + all_questions_raw[i]
        search_results.append(new_result)
        res_number += 1
    return search_results


### MAIN. Делаем первичную обработку данных, представляем их в удобном виде.
all_questions_raw, all_questions_processed, all_question_ids = sequencing_texts(getting_texts())

### Создаём матрицу TF-IDF.
tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(all_questions_processed)


### ### ###
### ПРИЛОЖЕНИЕ ДЛЯ ПОИСКОВИКА ###
### ### ###

app = Flask(__name__, static_url_path='/static')


@app.route('/')
def index():
    urls = {'Получить выдачу': url_for('result'),
            'Школа лингвистики НИУ ВШЭ': 'https://ling.hse.ru/'}
    return render_template('index.html', urls=urls)


@app.route('/', methods=['POST'])
def my_form_post():
    inserted_query_text = request.form['text']
    urls = {'На главную': url_for('index'),
            'Школа лингвистики НИУ ВШЭ': 'https://ling.hse.ru/'}
    result_list = searching_cosine_similarity(query_preprocessing(inserted_query_text))
    return render_template('result.html', urls=urls, result_list=result_list)


@app.route('/result')
def result():
    urls = {'На главную': url_for('index'),
            'Школа лингвистики НИУ ВШЭ': 'https://ling.hse.ru/'}
    return render_template('result.html', urls=urls)    
    

if __name__ == '__main__':
    app.run(debug=True)