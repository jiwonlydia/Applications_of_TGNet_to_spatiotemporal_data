import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'AppleGothic'
from datetime import * 
import requests
import json


# - 스마트서울 도시데이터 센서(S-DoT) 유동인구 측정정보
# - https://data.seoul.go.kr/dataList/OA-15964/S/1/datasetView.do

# # 유동인구 데이터 불러오기

# In[4]:


# train: 2022-01-01 00:00 ~ 2022-03-31 23:00 => 89일
# test: 2022-04-01 00:00 ~ 2022-04-30 23:00 => 29일
# print(date(2022,3,31) - date(2022,1,1))
# print(date(2022,4,30) - date(2022,4,1))

start = date(2021,12,27)

full_data = pd.DataFrame()
for i in range(0,130,7):
    try:
        s = start + timedelta(days=i)
        ss = s.strftime('%Y.%m.%d')    
        e = s + timedelta(days=6)
        ee = e.strftime('%m.%d')

        filename = 'S-DoT_WALK_' + str(ss) + '-' + str(ee) + '.csv'
        data = pd.read_csv('./sdot_data/' + filename, encoding='cp949')
        print(filename)
        full_data = pd.concat([full_data, data], axis=0)
    except:
        pass



def prep(df):
    df.reset_index(drop=True, inplace=True)
    df1 = df.iloc[:,:-2]
    df1.columns = df.columns[2:]
    df1 = df1[['시리얼', '날짜', '방문자수']]
    df1.rename({'시리얼':'사이트명'}, axis=1, inplace=True)
    df1['날짜'] = df1['날짜'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d%H%M'))

    return df1



df = prep(full_data)

# ### 위치정보 메타데이터
meta = pd.read_excel('./sdot_data/pop_location_meta.xlsx', header=2)
meta = meta[['사이트명','위도','경도']]

# train: 2022-01-01 00:00 ~ 2022-03-31 23:00 => 90일
# test: 2022-04-01 00:00 ~ 2022-04-30 23:00 => 30일

train_start = datetime(2022,1,1,0,0)
train_end = datetime(2022,3,31,23,50)

train_time_list = []
for i in range(90*24*6):
    t = train_start + timedelta(minutes=i*10)
    print(t)
    train_time_list.append(t)


test_start = datetime(2022,4,1,0,0)
test_end = datetime(2022,4,30,23,0)

test_time_list = []
for i in range(30*24*6):
    t = test_start + timedelta(minutes=i*10)
    print(t)
    test_time_list.append(t)


# train test split
df_train = df[(df['날짜'] >= train_start) & (df['날짜'] <= train_end)]
df_test = df[(df['날짜'] >= test_start) & (df['날짜'] <= test_end)]

temp = df_train.groupby(['사이트명']).count().reset_index()
site_list = temp.loc[temp['날짜']>=10000,'사이트명'].tolist()
df_train = df_train[df_train['사이트명'].isin(site_list)]


# 30*24*6 = 4320
temp = df_test.groupby(['사이트명']).count().reset_index()
site_list = temp.loc[temp['날짜']>=3000,'사이트명'].tolist()
df_test = df_test[df_test['사이트명'].isin(site_list)]


def fill_missing(df1, time_list):
    df2 = pd.DataFrame()

    for num in df1['사이트명'].unique():
        temp = df1[df1['사이트명']==num]
        df_time = pd.DataFrame({'날짜':time_list})
        temp2 = pd.merge(temp, df_time, on='날짜', how='right')
        temp2['사이트명'] = temp2['사이트명'].fillna(num)
        temp2 = temp2.drop_duplicates(['날짜'], keep='first')

        # 시계열 보간법으로 방문자수 결측값 채우기
        try:
            temp2['방문자수'] = pd.Series(temp2['방문자수'].tolist(), index=temp2['방문자수']).interpolate(method='time').tolist()
        except:
            temp2['방문자수'] = temp2['방문자수'].ffill()
#             temp2['방문자수'] = temp2['방문자수'].bfill()

        temp2['방문자수'] = temp2['방문자수'].ffill()
        temp2['방문자수'] = temp2['방문자수'].bfill()
        df2 = pd.concat([df2, temp2], axis=0)
    df2 = df2.dropna()
    return df2


# 결측치가 존재한다면 보간법으로 채우기
df_train1 = fill_missing(df_train, train_time_list)
df_test1 = fill_missing(df_test, test_time_list)


# ## 30분 단위로 subsampling
df_train2 = df_train1[(df_train1['날짜'].dt.minute == 0)|(df_train1['날짜'].dt.minute == 30)]
df_test2 = df_test1[(df_test1['날짜'].dt.minute == 0)|(df_test1['날짜'].dt.minute == 30)]

# ## 위치정보와 합치기

df_train3 = pd.merge(df_train2, meta, on=['사이트명'], how='left')
df_train3 = df_train3.dropna()

df_test3 = pd.merge(df_test2, meta, on=['사이트명'], how='left')
df_test3 = df_test3.dropna()

df_train3.to_csv('df_train_pop.csv', index=False)
df_test3.to_csv('df_test_pop.csv', index=False)


max_lat = df_train3['위도'].max() ; print(max_lat)
min_lat = df_train3['위도'].min() ; print(min_lat)
unit_lat = (max_lat - min_lat)/10 ; print(unit_lat)

max_long = df_train3['경도'].max() ; print(max_long)
min_long = df_train3['경도'].min(); print(min_long)
unit_long = (max_long - min_long)/20 ; print(unit_long)


lat_list = []
lat = min_lat
for _ in range(11):
    lat_list.append(lat)
    lat += unit_lat
print('lat_list', lat_list)

long_list = []
long = min_long
for _ in range(21):
    long_list.append(long)
    long += unit_long
print('long_list', long_list)


df_train = df_train3.copy()
num_days = 90
num_30mins = 2 * 24 * num_days

train_image = np.zeros(shape=(num_30mins,10,20))

for k in range(num_30mins):
    target_datetime = train_start + timedelta(minutes=k*30)
    print(target_datetime)
    tmp = df_train[df_train['날짜'] == target_datetime]
    
    for i in range(10):
        a = (tmp['위도'] >= lat_list[i]) & (tmp['위도'] <= lat_list[i+1])
        
        for j in range(20):
            b = (tmp['경도'] >= long_list[j]) & (tmp['경도'] <= long_list[j+1])
            if len(tmp.loc[a&b,'방문자수']) > 0:
                val = tmp.loc[a&b,'방문자수'].mean()
                train_image[k][i][j] = val


df_test = df_test3.copy()
num_days = 30
num_30mins = 2 * 24 * num_days

test_image = np.zeros(shape=(num_30mins,10,20))

for k in range(num_30mins):
    target_datetime = test_start + timedelta(minutes=k*30)
    print(target_datetime)
    tmp = df_test[df_test['날짜'] == target_datetime]
    
    for i in range(10):
        a = (tmp['위도'] >= lat_list[i]) & (tmp['위도'] <= lat_list[i+1])
        
        for j in range(20):
            b = (tmp['경도'] >= long_list[j]) & (tmp['경도'] <= long_list[j+1])
            if len(tmp.loc[a&b,'방문자수']) > 0:
                val = tmp.loc[a&b,'방문자수'].mean()
                test_image[k][i][j] = val


print('train_image', train_image.shape)
print('test_image', test_image.shape)


# 이미지 회전
train_image_new = train_image.copy()
for i in range(10):
    train_image_new[:,i,:] = train_image[:,9-i,:]


train_image = train_image_new.copy()

# heatmap image 시각화
# vmax = train_image[0:48].max() # 252
# vmax = 200
# c = 30

# f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(20,10))

# ax1.imshow(train_image[0][:][:] + c, cmap='gray', vmin=0, vmax=vmax) # 00:00
# ax1.set_title('2022-01-01 00:00', fontsize=20)

# ax2.imshow(train_image[6][:][:] + c, cmap='gray', vmin=0, vmax=vmax) # 03:00
# ax2.set_title('2022-01-01 03:00', fontsize=20)

# ax3.imshow(train_image[12][:][:] + c, cmap='gray', vmin=0, vmax=vmax) # 06:00
# ax3.set_title('2022-01-01 06:00', fontsize=20)

# ax4.imshow(train_image[18][:][:] + c, cmap='gray', vmin=0, vmax=vmax) # 09:00
# ax4.set_title('2022-01-01 09:00', fontsize=20)

# ax5.imshow(train_image[24][:][:] + c, cmap='gray', vmin=0, vmax=vmax) # 12:00
# ax5.set_title('2022-01-01 12:00', fontsize=20)

# ax6.imshow(train_image[30][:][:] + c, cmap='gray', vmin=0, vmax=vmax) # 15:00
# ax6.set_title('2022-01-01 15:00', fontsize=20)

# ax7.imshow(train_image[36][:][:] + c, cmap='gray', vmin=0, vmax=vmax) # 18:00
# ax7.set_title('2022-01-01 18:00', fontsize=20)

# ax8.imshow(train_image[42][:][:] + c, cmap='gray', vmin=0, vmax=vmax) # 21:00
# ax8.set_title('2022-01-01 21:00', fontsize=20)

# ax9.imshow(train_image[48][:][:] + c, cmap='gray', vmin=0, vmax=vmax) # 24:00
# ax9.set_title('2022-01-02 00:00', fontsize=20)

# plt.tight_layout()
# plt.show()



np.save('./data_preprocessed/train_image_pop', train_image)
np.save('./data_preprocessed/test_image_pop', test_image)

