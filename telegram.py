import telebot
from client import VKClient
import markups

from datetime import datetime

# import telegram

from telebot import types

bot = telebot.TeleBot('5321818418:AAHS5Ret8iJ2BQH7N3G6vuOzkyTPGXBLnI0')


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id,"Введи username.")


vk_client = VKClient()
posts = []


def analysis(text):
    result = 'neutral'
    if text[:6] == 'Друзья':
        result = 'positive'

    if result == 'neutral':
        return 'Нейтральные эмоции'
    elif result == 'positive':
        return 'Позитивные эмоции'
    elif result == 'negative':
        return 'Негативные эмоции'
    else:
        return ''


# text getter
@bot.message_handler(content_types=['text'])
def get_text_messages(message):

    try:
        index = int(message.text) - 1
        if 0 <= index < 10:
            posts, links, dates = vk_client.get_posts()
            analysis_result = analysis(posts[index])
            bot.send_message(message.from_user.id, analysis_result)
            bot.send_message(message.from_user.id, 'Какой пост анализируем?')
        return
    except:
        pass


    username = message.text

    posts, links, dates = vk_client.get_posts(username)


    # bot.send_message(chat_id=message.chat.id, parse_mode='HTML',
    #                       text=posts,
    #                       reply_markup=markups.make_basket(posts))


    for i in range(len(posts)):
        bot.send_message(message.from_user.id, 'Пост '+str(i+1)+':\n------------------------------------------------------------------------------------------------\n'+posts[i][:min(len(posts[i]), 250)] + '\n------------------------------------------------------------------------------------------------\nПодробнее: '+ links[i] + '\nДата публикации: '+ str(datetime.utcfromtimestamp(dates[i]).strftime('%Y-%m-%d %H:%M:%S'))) 


    bot.send_message(message.from_user.id, 'Какой пост анализируем?')



bot.polling(none_stop=True, interval=0)
