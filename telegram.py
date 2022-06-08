import telebot
from client import VKClient
import markups
from datetime import datetime
from telebot import types
import vars
import csv

from roberta_rus.roberta_predict import predict


TOKEN = '5321818418:AAHS5Ret8iJ2BQH7N3G6vuOzkyTPGXBLnI0'

bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id,"Введи username.")


vk_client = VKClient()

def get_analysis(text):
    result = predict(text)
    if result == 'neutral':
        return 'нейтральные эмоции'
    elif result == 'positive':
        return 'позитивные эмоции'
    elif result == 'negative':
        return 'негативные эмоции'
    else:
        return ''


def edit_message(call):
    date = str(datetime.utcfromtimestamp(dates[vars.NUMBER_OF_POST]).strftime('%Y-%m-%d %H:%M:%S'))
    link = links[vars.NUMBER_OF_POST]
    analysis_result = analysis[vars.NUMBER_OF_POST]
    text1 = posts[vars.NUMBER_OF_POST][:100]

    bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id, parse_mode='HTML',
                      text=text1,
                      reply_markup=markups.make_product(date, link, analysis_result))


@bot.callback_query_handler(func=lambda call: call.data == '1')
def get_all_user_posts(call):

    global USERNAME
    global posts, links, dates

    vars.COUNT_POSTS = len(posts)
    vars.NUMBER_OF_POST = 0

    edit_message(call)


@bot.callback_query_handler(func=lambda call: call.data == '20')
def query_handler(call):
    vars.NUMBER_OF_POST -= 1
    if vars.NUMBER_OF_POST == -1:
        vars.NUMBER_OF_POST = vars.COUNT_POSTS - 1

    edit_message(call)


@bot.callback_query_handler(func=lambda call: call.data == '30')
def query_handler(call):
    vars.NUMBER_OF_POST += 1
    if vars.NUMBER_OF_POST >= vars.COUNT_POSTS:
        vars.NUMBER_OF_POST = 0

    edit_message(call)


@bot.callback_query_handler(func=lambda call: call.data == '60')
def get_all_user_posts(call):

    global posts, links, dates, analysis

    analysis_result = get_analysis(posts[vars.NUMBER_OF_POST])

    analysis[vars.NUMBER_OF_POST] = analysis_result

    edit_message(call)


@bot.message_handler(content_types=['text'])
def get_text_messages(message):

    global USERNAME
    USERNAME = message.text

    global posts, links, dates, analysis
    posts, links, dates = vk_client.get_posts(USERNAME)

    analysis = [''] * len(posts)

    bot.send_message(message.chat.id, 'Пользователь: @' + USERNAME,
                         reply_markup=markups.catalog)


@bot.callback_query_handler(func=lambda call: call.data == '80')
def query_handler(call):

    global USERNAME

    bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id, parse_mode='HTML',
        text='Пользователь: @' + USERNAME,
                         reply_markup=markups.catalog)


@bot.callback_query_handler(func=lambda call: call.data == '2')
def get_history(call):

    global posts, links, dates, analysis

    csv_data = []

    for i in range(len(posts)):
        if analysis[i]:
            csv_data.append([posts[i], links[i], dates[i], analysis[i]])

    header = ['post', 'link', 'date', 'analisys']

    with open('results.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_data)


    file = 'results.csv'
    doc = open(file, 'rb')

    bot.send_document(call.message.chat.id, doc)



bot.polling(none_stop=True, interval=0)
