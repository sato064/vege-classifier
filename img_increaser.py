#反転画像を保存する処理

import cv2
import os
import time
import datetime

tm_start = time.time()            #処理時間計測用
dt_now = datetime.datetime.now()  # 現在日時
dt_date_str = dt_now.strftime('%Y/%m/%d %H:%M')
print(dt_date_str)

path = 'imgs/eggplant'

files = os.listdir(path)

files_file =  [f for f in files if os.path.isfile(os.path.join(path, f))]

for y_pic in files_file:

  img = cv2.imread(path + "/" + y_pic)

  y = cv2.flip(img,1)

  cv2.imwrite(path + "/y_" + y_pic,y)

tm_end = time.time()
print('処理完了')
print('------------------------------------')
total = tm_end - tm_start
total_str = f'トータル時間: {total:.1f}s({total/60:.2f}min)'
print(total_str)

tm_start = time.time()            #処理時間計測用
dt_now = datetime.datetime.now()  # 現在日時
dt_date_str = dt_now.strftime('%Y/%m/%d %H:%M')
print(dt_date_str)

files = os.listdir(path)

files_file =  [f for f in files if os.path.isfile(os.path.join(path, f))]

print(files_file)


for y_pic in files_file:

  img = cv2.imread(path + '/' + y_pic)

  img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  cv2.imwrite(path + '/gray_' + y_pic,img_gray)

tm_end = time.time()
print('処理完了')
print('------------------------------------')
total = tm_end - tm_start
total_str = f'トータル時間: {total:.1f}s({total/60:.2f}min)'
print(total_str)