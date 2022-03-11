from nltk.tag import pos_tag
import nltk
import pymorphy2

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
sentence = " В рамках программы сотрудничества заключено соглашение о " \
           "закупке для авиакомпании Cairo Aviation (дочернее предприятие KATO Investment ) " \
           "6 самолетов МС-21.  Российская корпорация 'Иркут' и египетский " \
           "холдинг KATO Investment на авиасалоне в Дубае " \
           "подписали ряд соглашений о развитии сотрудничества" \
           " по проекту новейшего российского пассажирского самолета МС-21 "
tagged_sent = pos_tag(sentence.split())
propernouns = [word for word,pos in tagged_sent if pos == 'NNP' or pos == "NN" or pos == 'POS']
propernouns = list(set(list(filter(lambda x : x[0] == x[0].upper(), propernouns))))
prob_thresh = 1

morph = pymorphy2.MorphAnalyzer()

result = []

for word in nltk.word_tokenize(sentence):
    for p in morph.parse(word):
        if p.score >= prob_thresh and word in propernouns:
                result.append(word)

print(set(result))
