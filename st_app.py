import csv
import numpy as np
import streamlit as st
import pandas
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

colors = [
    '#bbbbbb', '#a8328b', '#7332a8', '#5e4fa2', '#3288bd', '#66c2a5', '#abdda4', '#fdae61', '#f46d43', '#d53e4f', '#9e0142',
]

#получение данных
st_tickers_data = pandas.read_csv('cache/tickers_data.csv', header = 1, index_col=0)
st_caps = pandas.read_csv('cache/caps.csv', header = None, index_col = 0)
with open('cache/pie.csv', mode='r') as csv_file:
  csv_reader = csv.reader(csv_file)
  pie_csv = list(csv_reader)

st.title('Анализатор акций рейтинга S&P500')
st.markdown('# Описание проекта')
st.markdown('''- [x] Обработка данных с помощью pandas.
- [x] Веб-скреппинг.
- [x] Работа с REST API (XML/JSON).
- [x] Визуализация данных.
- [x] Математические возможности Python (содержательное использование numpy/scipy, SymPy и т.д. для решения математических задач).
- [x] Streamlit.
- [x] SQL.
- [x] Работа с геоданными с помощью geopandas, shapely, folium и т.д.
- [x] Машинное обучение (построение предсказательных моделей типа регрессий или решающих деревьев).
- [x] Дополнительные технологии (библиотеки, не обсуждавшиеся в ходе курса — например, телеграм-боты, нейросети или ещё что-нибудь).''')
st.markdown('[Github repository](https://github.com/ashmyrina/Final)')

st.markdown('## Визулизация данных')
st.markdown('Котировки акций, входящих в рейтинг S&P500')
st.write(st_tickers_data.iloc[1:, :503])

# --------------------------------------------------------------------------- #
st.markdown('### Круговая диаграмма капитализаций компаний рейтинга S&P500')
SLICE_NUM = 10
pie_l = pie_csv[0]
pie_x = [float(x) for x in pie_csv[1]]

colors = [
    '#bbbbbb', '#a8328b', '#7332a8', '#5e4fa2', '#3288bd', '#66c2a5', '#abdda4', '#fdae61', '#f46d43', '#d53e4f', '#9e0142',
]

fig, ax = plt.subplots()
wedges, texts, percs = ax.pie(pie_x, labels=pie_l, startangle=90, autopct='%1.1f%%', pctdistance=0.82, colors=colors)
dev = [autotext.set_color('white') for autotext in percs]

radfraction = 0.02
group = list(range(1, SLICE_NUM + 1))
ang = np.deg2rad((wedges[group[-1]].theta2 + wedges[group[0]].theta1) / 2)
for j in group:
    center = radfraction * wedges[j].r * np.array([np.cos(ang), np.sin(ang)])
    wedges[j].set_center(center)
    text_offset = np.array(texts[j].get_position()) + center
    texts[j].set_position(1 * text_offset)
    percs[j].set_position(np.array(percs[j].get_position()) + center)
ax.autoscale(True)
st.pyplot(fig)

# Отрисовка цен и объемов акций по времени ---------------------------------- #
X_TICKS_NUM = 7
idx = np.round(np.linspace(1, len(st_tickers_data) - 2, X_TICKS_NUM)).astype(int)
x_labels = [x[:7] for x in st_tickers_data.index[idx]]
x_ticks = st_tickers_data.index[idx]

st.markdown('### Изменения цен акций')
fig = plt.figure(figsize = (16, 5))
for idx, x in enumerate(st_caps[-10:].index):
  plt.plot(st_tickers_data.index, st_tickers_data[x], label = x, color=colors[1:][idx])
plt.xticks(x_ticks, labels=x_labels)
plt.xlabel("Дата")
plt.ylabel('Цена продажи')
plt.legend()
plt.grid(True)
st.pyplot(fig)

st.markdown('### Изменения количества торгов')
fig = plt.figure(figsize = (16, 5))
for idx, x in enumerate(st_caps[-10:].index):
  plt.plot(st_tickers_data.index, st_tickers_data[f'{x}.5'], label = x, color=colors[1:][idx])
plt.xticks(x_ticks, labels=x_labels)
plt.xlabel("Дата")
plt.ylabel('Количество сделок по продаже акций')
plt.legend()
plt.grid(True)
st.pyplot(fig)

# Скользящее среднее подсчитанно с помощью соответствующей оконной функции
window_size = 5
st.markdown(f'### Скользящее среднее объема торгов за {window_size} дней')
fig = plt.figure(figsize = (16, 5))
for idx, x in enumerate(st_caps[-10:].index):
  plt.plot(st_tickers_data.index, st_tickers_data[f'{x}.5'].rolling(window_size).agg(lambda r: r.mean()), label = x, color=colors[1:][idx])
plt.xticks(x_ticks, labels=x_labels)
plt.xlabel("Дата")
plt.ylabel('Количество сделок по продаже акций')
plt.legend()
plt.grid(True)
st.pyplot(fig)

# --------------------------------------------------------------------------- #
st.markdown('## Географическое распределение капитала')

map_data = []
with open('cache/map.csv', mode='r') as csv_file:
  csv_reader = csv.reader(csv_file)
  line_count = 0

  map_data = list(csv_reader)  

x = [float(d) for d in map_data[0]]
y = [float(d) for d in map_data[1]]
z = [float(d) for d in map_data[2]]

worldmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
fig, ax = plt.subplots(figsize=(12, 7))
worldmap.plot(color="lightgrey", ax=ax)

cbar_colors = [
    (0.251, 0.224, 0.565),
    (0.439, 0.776, 0.635),
    (0.902, 0.945, 0.273),
    (0.992, 0.859, 0.498),
    (0.957, 0.427, 0.271),
    (0.763, 0.09, 0.271)
]
cbar_map = LinearSegmentedColormap.from_list('custom', cbar_colors)

plt.scatter(x, y, c=z, s=z, alpha=0.6, vmin=min(z), vmax=max(z),cmap=cbar_map)
plt.colorbar(label='Объём капитализации, $ МЛРД', fraction=0.023, pad=0.04)
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.show()
st.pyplot(fig) # Карта
# --------------------------------------------------------------------------- #

st.markdown('## Прогнозирование стоимости акций')

image = Image.open('cache/corr.png')
st.image(image, caption='Sunrise by the mountains')

# --------------------------------------------------------------------------- #
st.markdown('## Регрессия')
st.latex(r'price_{\tau-n}, ..., price_{\tau}) \rightarrow price_{\tau+1}')
st.latex(r'price_{\tau-n}, ..., price_{\tau}, volume_{\tau-n}, ..., volume_{\tau}) \rightarrow price_{\tau+1}')

st.markdown('Первая модель учитывает только предыдущие *n* измерений, вторая же будет дополнятся еще и соответствующими параметрами *volume*.')

with open('cache/regression.csv', 'r') as csv_file:
  csv_reader = csv.reader(csv_file)
  regression_data = list(csv_reader)

y_pred_vol = [float(y) for y in regression_data[0]]
y_pred = [float(y) for y in regression_data[1]]
y_true = [float(y) for y in regression_data[2]]

fig = plt.figure(figsize = (7, 7))
plt.plot(y_pred_vol, label = 'y_pred_vol')
plt.plot(y_pred, label = 'y_pred')
plt.plot(y_true, label = 'y_true')

plt.legend()
plt.ylabel('Стоимость акции, $')
plt.xlabel('Номер точки')
plt.grid(True)

st.pyplot(fig) # График
# --------------------------------------------------------------------------- #
