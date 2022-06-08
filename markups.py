from telebot import types
import vars

catalog = types.InlineKeyboardMarkup(row_width=5)

catalog.add(types.InlineKeyboardButton('Получить посты пользователя', callback_data=1))
catalog.add(types.InlineKeyboardButton('Получить историю анализа пользователя', callback_data=2))


def make_product(date, link, analysis_result):
    product = types.InlineKeyboardMarkup(row_width=1)

    product.add(types.InlineKeyboardButton('{}'.format(date), callback_data=11))
    btn1 = types.InlineKeyboardButton('⬅', callback_data=20)
    btn2 = types.InlineKeyboardButton('{}/{}'.format(vars.NUMBER_OF_POST + 1, vars.COUNT_POSTS),
                                      callback_data=25)
    btn3 = types.InlineKeyboardButton('➡', callback_data=30)
    product.row(btn1, btn2, btn3)
    product.add(types.InlineKeyboardButton('Подробнее', url=link, callback_data=50))
    if not analysis_result:
        product.add(types.InlineKeyboardButton('Анализировать пост', callback_data=60))
    else:
        product.add(types.InlineKeyboardButton('Результат анализа: ' + analysis_result, callback_data=70))
    product.add(types.InlineKeyboardButton('Назад', callback_data=80))

    return product