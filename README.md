[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fgjbae1212%2Fhit-counter)](https://hits.seeyoufarm.com)                    

# 제주도 사용금액 데이터를 통한 소비행태 및 재난지원금 효과 분석 :moneybag:

## 개요 

### 1.1  프로젝트 주제
- __“공간정보를 활용한 탐색적 데이터 분석”__ 이라는 주제로 2020년 5월~8월 제주지역 시간별, 업종별, 지역별 소비금액 및 재난지원금의 소비형태를 다양한 시각으로 분석하고 시각화하고자 합니다.
- 본 프로젝트는 국토연구원이 주최하고 DACON이 주관하는 __"공간 정보를 활용한 탐색적 데이터분석 경진대회"__ 에 참여한 프로젝트입니다. 
- https://dacon.io/competitions/official/235682/overview/

### 1.2 프로젝트 목적
#### 1) 제주도의 전반적인 소비행태 확인
- 제주도는 관광산업이 매우 발달한 지역입니다. 또한 특정 지역을 제외하면 관광지와 주거지에 구분이 비교적 명확한 편입니다. 이에 <u>지역별/ 업종별 소비 행태</u>를 파악하고 이를 통해 지역적 특성을 파악하고자 합니다.
#### 2) 재난지원금이 소비자와 소상공인, 자영업자 등 “서민" 경제에 도움을 주었는지 확인
- 재난지원금은 소상공인ㆍ자영업자 등을 지원하고, 사회취약계층에 대한 사각지대를 살피는 등 코로나19의 위기로부터 국민 생활의 안정과 위축된 경제 회복을 위함에 목적이 있습니다.(행정안전부 발췌)
- 이에 제주지역의 재난지원금 사용 행태를 기간(월/시간)별, 지역별, 업종별, 업종 규모별로 확인하고 <u>실제 재난지원금이 소상공인과 자영업자, 서민들의 가계 경제에 도움이 되었는지를 확인</u>해보고자 합니다.
- 분석을 통해 <u> 재난지원금의 정책은 의도대로 진행되었는지 </u>, 의도와는 다른 양상의 소비행태를 보였는지 알아보고자 합니다.

### 1.3 프로젝트 진행순서
- 환경설정 및 데이터 전처리를 진행합니다.
- 전반적인 data skimming을 시작으로 기간별(월/시간), 업종별, 업종규모별, 지역별 등으로 세분화하여 데이터 시각화를 진행하였습니다.
- 업종의 종류가 다양하여 각 기준별 상위 10개 업종 또는 자체적으로 구분한 카테고리별 분포를 집중적으로 확인하였습니다. 
 
### 1.4 시작에 앞서
- 본 프로젝트를 진행하기 위해서는 __Python 3__ 이상의 버젼과 다음의 설치가 필요합니다.
```
pip install glob
pip install numpy
pip install pandas
pip install seaborn
pip install matplotlib
pip install pickle
pip install warnings
pip install folium

```
- 자세한 코드는 '3. results/jeju_consumption_eda.ipynb' 파일을 참고해주시기 바랍니다.


## 전처리

### 1.1 환경설정
```python3
from glob import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
import warnings
warnings.filterwarnings(action='ignore')
import gzip

from matplotlib import font_manager
# 한글설정(MAC)
f_path = '/Library/Fonts/NanumGothic.ttf'
font_manager.FontProperties(fname=f_path).get_name()
from matplotlib import rc
rc('font', family = 'NanumGothic')

# 한글 설정 (WIN)
# from matplotlib import rc
# plt.rcParams['axes.unicode_minus'] = False
# f_path= "C:/Windows/Fonts/malgun.ttf"
# font_name= font_manager.FontProperties(fname=f_path).get_name()
# rc('font', family =font_name)
# plt.rc('font', family='Malgun Gothic')

import folium
from folium.plugins import MarkerCluster, MiniMap

pd.options.display.float_format = '{:.5f}'.format
```

### 1.1 데이터 주소 가져오기
- 주어진 ITRF 위,경도 주소 -> wgs84좌표 변환
```python
# 1. 좌표 변환 하기 
df_6 = raw_data_6.copy()

from pyproj import Proj, transform
# ITRF좌표계
proj_ITRF = Proj(init='epsg:5179')
# WGS 좌표계 
proj_WGS84 = Proj(init='epsg:4326')


df_6['lon'], df_6['lat'] = transform(proj_ITRF, proj_WGS84, df_6['POINT_X'], df_6['POINT_Y'])
```
<img src="https://user-images.githubusercontent.com/72846894/103287917-07cd3100-4a27-11eb-8684-dda5a66282da.png"></img>

- wgs84좌표 변환 -> 주소 변환 -> 읍면동 데이터 가져오기 (카카오API 활용)
```python
import requests
import json

dong_ls = []

for i in range(len(df_uniq.index)):
    url = df_uniq['url'][i]
    headers = {"Authorization": "<REST API KEY>"}
    api_test = requests.get(url, headers=headers)
    url_text = json.loads(api_test.text)
    dong = url_text['documents'][0]['region_3depth_name']
    dong_ls.append(dong)
print('converted point to addr')

# 가져온 주소값 df 변환 
dic = {'dong':dong_ls}
df_dong = pd.DataFrame(dic)
```
<img src="https://user-images.githubusercontent.com/72846894/103288058-61356000-4a27-11eb-8b41-22b2a17c3a94.png" width="70%" height="80%"></img>


### 1.2 데이터 "Type" 열 분류작업
- Type(업종)이 200여개로 분류되어있어 유사업종을 하나의 카테고리로 병합하는 작업을 하였습니다. 
- 19개의 카테고리로 병합, 'Category' column 생성
```python

jeju_cat.replace(dict.fromkeys(['택시', '기타교통수단', '통신기기'], '교통/통신'), inplace=True)
jeju_cat.replace(dict.fromkeys(['대형할인점', '농축협직영매장', '농협하나로클럽', '편의점', '슈퍼마켓', '정육점', '기타유통업', '연쇄점', '상품권', '인삼제품', '홍삼제품', '농축수산품', '주류판매점', '기타건강식', '건강식품(회원제형태)'], '마트/편의점(유통)'), inplace=True)
jeju_cat.replace(dict.fromkeys(['악기점', 'DVD음반테이프판매', '문화취미기타', '영화관', '티켓', '수족관', '화랑'], '영화/문화'), inplace=True)
jeju_cat.replace(dict.fromkeys(['내의판매점', '양품점', '옷감직물', '귀금속', '기타직물', '스포츠의류', '가방', '정장', '악세사리', '기타잡화', '신발', '기타의류', '단체복', '아동의류', '캐쥬얼의류', '맞춤복점', '제화점', '인터넷Mall', '인터넷종합Mall'], '쇼핑/패션'), inplace=True)
jeju_cat.replace(dict.fromkeys(['제과점', '스넥', '기타음료식품'], '카페/베이커리'), inplace=True)
jeju_cat.replace(dict.fromkeys(['미용재료', '화장품', '피부미용실', '미용원', '이용원'], '뷰티'), inplace=True)
jeju_cat.replace(dict.fromkeys(['구내매점', '단란주점', '주점', '서양음식', '일반한식', '일식회집', '중국음식',  '유흥주점', '칵테일바'], '외식/주점'), inplace=True)
jeju_cat.replace(dict.fromkeys(['세탁소', '애완동물', '화원', '침구수예점', '성인용품점', '가전제품', '기타가구', '조명기구', '민예공예품', '주방용식기', '기타전기제품', '소프트웨어', '주방용구', '카페트커텐천막', '컴퓨터', '카메라', '일반가구', '정수기', '철제가구', '시계', '안경', '사우나', '안마스포츠마사지', '사진관', '인테리어', 'CATV'], '생활/기타'), inplace=True)
jeju_cat.replace(dict.fromkeys(['완구점', '화방표구점', '문구용품', '일반서적', '출판인쇄물', '기타사무용', '사무기기', '전문서적', '정기간행물', '기타서적문구', '서적출판(회원제형태)'], '서점/문구'), inplace=True)
jeju_cat.replace(dict.fromkeys(['기타보험', '손해보험'], '금융'), inplace=True)
jeju_cat.replace(dict.fromkeys(['기념품점', '기타숙박업', '특급호텔', '2급호텔', '1급호텔', '콘도', '렌트카', '관광여행', '항공사', '여객선', '면세점'], '여행/숙박'), inplace=True)
jeju_cat.replace(dict.fromkeys(['레져용품수리', '스포츠레져용품', '골프용품', '노래방', '당구장', '골프경기장', '볼링장', '골프연습장', '헬스크럽', '레져업소(회원제형태)', '기타레져업', '종합레져타운', '수영장', '테니스장', '기타회원제형태업소', ], '레저/스포츠'), inplace=True)
jeju_cat.replace(dict.fromkeys(['유아원', '독서실', '보습학원', '기능학원', '기타교육', '외국어학원', '학원(회원제형태)', '예체능학원', '컴퓨터학원', '대학등록금', '학습지교육', '초중고교육기관'], '교육/육아'), inplace=True)
jeju_cat.replace(dict.fromkeys(['주유소', '자동차정비', 'LPG', '세차장', '유류판매', '이륜차판매', '윤활유전문판매', '자동차시트타이어', '중고자동차', '수입자동차', '자동차부품', '주차장', '기타자동차서비스', '카인테리어', ], '주유/자동차'), inplace=True)
jeju_cat.replace(dict.fromkeys(['약국', '종합병원', '기타의료기관및기기', '한약방', '제약회사', '의료용품', '건강진단', '한의원', '동물병원', '의원', '치과의원', '병원', '산후조리원', '치과병원'], '의료'), inplace=True)
jeju_cat.replace(dict.fromkeys(['목재석재철물', '건축요업품', '골동품점', '기계공구', '기타건축자재', '보일러펌프', '페인트', '냉열기기', '유리', '과학기자재', '기타광학품', '기타연료', '중장비수리', ], '건설/제조'), inplace=True)
jeju_cat.replace(dict.fromkeys(['기타농업관련', '비료농약사료종자', '농기계', '미곡상'], '농업'), inplace=True)
jeju_cat.replace(dict.fromkeys(['사무서비스', '기타대인서비스', '화물운송', '보관창고업', '종합용역', '조세서비스', '가례서비스', '공공요금', '위탁급식업', '기타용역서비스', '기타수리서비스', '기타운송', '가정용품수리', '견인서비스', '부동산중개임대', '부동산분양', '신변잡화수리', '사무통신기기수리', '법률회계서비스', '사무서비스(회원제형태)', '정보서비스', '기타업종', '기타비영리유통'], '서비스/기타'), inplace=True)
```


## EDA

### 1. 제주도 소비 전반적 시각화 
#### 1) 기간별/지역별/업종별 시각화
```python3
# 총사용금액으로 각 컬럼별 데이터 분석 

figure, ((ax1,ax2), (ax3,ax4),(ax5,ax6)) = plt.subplots(nrows=3, ncols=2)
figure.set_size_inches(30,8)

sns.barplot(data=jeju_all, x="YM", y="TotalSpent", ax=ax1)  #월별 총사용금액 
sns.barplot(data=jeju_all, x="Time", y="TotalSpent", ax=ax2)  #시간대별 총 사용금액 
sns.barplot(data=jeju_all, x="SIGUNGU", y="TotalSpent", ax=ax3) #제주/서귀포 총 사용금액
sns.barplot(data=jeju_all, x="dong", y="TotalSpent", ax=ax4) # 읍면동별 총 사용금액 
sns.barplot(data=jeju_all, x="Type", y="TotalSpent", ax=ax5) # 업종별 사용금액 
sns.barplot(data=jeju_all, x="Category", y="TotalSpent", ax=ax6) # 업종구분별 사용금액 
```
<img src="https://user-images.githubusercontent.com/72846894/103288940-5da2d880-4a29-11eb-8f08-e3888ca363af.png"></img>
<img src="https://user-images.githubusercontent.com/72846894/103288974-74492f80-4a29-11eb-8054-81240bc38536.png"></img>
<img src="https://user-images.githubusercontent.com/72846894/103288999-832fe200-4a29-11eb-8e7b-d41da2f93db3.png"></img>


#### 2) 월별 소비 상위 10개 업종 분석(이용건수 기준)
```python3
# 월별 업종별로 그룹핑하여 평균 구하기
jeju_type = jeju_all.groupby(['YM', 'Type'], as_index=False).mean()

# 총 이용건수와 총 재난지원금 이용건수 컬럼의 정규화 과정
## 월별로 각 컬럼의 최대값으로 나누어 줌

col = ['NumofSpent', 'NumofDisSpent']
num_jeju_type_5 = (jeju_type[jeju_type['YM'] == 202005][col]) / (jeju_type[jeju_type['YM'] == 202005][col].max())
num_jeju_type_6 = (jeju_type[jeju_type['YM'] == 202006][col]) / (jeju_type[jeju_type['YM'] == 202006][col].max())
num_jeju_type_7 = (jeju_type[jeju_type['YM'] == 202007][col]) / (jeju_type[jeju_type['YM'] == 202007][col].max())
num_jeju_type_8 = (jeju_type[jeju_type['YM'] == 202008][col]) / (jeju_type[jeju_type['YM'] == 202008][col].max())

norm_jeju_type_5 = jeju_type[jeju_type['YM'] == 202005].copy()
norm_jeju_type_6 = jeju_type[jeju_type['YM'] == 202006].copy()
norm_jeju_type_7 = jeju_type[jeju_type['YM'] == 202007].copy()
norm_jeju_type_8 = jeju_type[jeju_type['YM'] == 202008].copy()

norm_jeju_type_5[col] = num_jeju_type_5[col]
norm_jeju_type_6[col] = num_jeju_type_6[col]
norm_jeju_type_7[col] = num_jeju_type_7[col]
norm_jeju_type_8[col] = num_jeju_type_8[col]

```

##### 2-1) 총이용건수 상위 10개 업종
```python
fig = plt.figure(figsize=(20,20))
ax1 = fig.add_subplot(411)
ax1 = sns.barplot(x='Type', y='NumofSpent', data=norm_jeju_type_5.nlargest(10, 'NumofSpent'), palette='coolwarm')
ax1.set_title('5월 업종별 평균 이용건수 (NumofSpent)')

ax2 = fig.add_subplot(412)
ax2 = sns.barplot(x='Type', y='NumofSpent', data=norm_jeju_type_6.nlargest(10, 'NumofSpent'), palette='coolwarm')
ax2.set_title('6월 업종별 평균 이용건수 (NumofSpent)')

ax3 = fig.add_subplot(413)
ax3 = sns.barplot(x='Type', y='NumofSpent', data=norm_jeju_type_7.nlargest(10, 'NumofSpent'), palette='coolwarm')
ax3.set_title('7월 업종별 평균 이용건수 (NumofSpent)')

ax4 = fig.add_subplot(414)
ax4 = sns.barplot(x='Type', y='NumofSpent', data=norm_jeju_type_8.nlargest(10, 'NumofSpent'), palette='coolwarm')
ax4.set_title('8월 업종별 평균 이용건수 (NumofSpent)')
plt.show()
```
<img src="https://user-images.githubusercontent.com/72846894/103289493-b3c44b80-4a2a-11eb-8057-23cabd9ca085.png"></img>

##### 2-2) 재난지원금 이용건수 상위 10개 업종
```python
fig = plt.figure(figsize=(20,20))
ax1 = fig.add_subplot(411)
ax1 = sns.barplot(x='Type', y='NumofDisSpent', data=norm_jeju_type_5.nlargest(10, 'NumofDisSpent'), palette='coolwarm')
ax1.set_title('5월 업종별 평균 지원금 이용건수 (NumofDisSpent)')

ax2 = fig.add_subplot(412)
ax2 = sns.barplot(x='Type', y='NumofDisSpent', data=norm_jeju_type_6.nlargest(10, 'NumofDisSpent'), palette='coolwarm')
ax2.set_title('6월 업종별 평균 지원금 이용건수 (NumofDisSpent)')

ax3 = fig.add_subplot(413)
ax3 = sns.barplot(x='Type', y='NumofDisSpent', data=norm_jeju_type_7.nlargest(10, 'NumofDisSpent'), palette='coolwarm')
ax3.set_title('7월 업종별 평균 지원금 이용건수 (NumofDisSpent)')

ax4 = fig.add_subplot(414)
ax4 = sns.barplot(x='Type', y='NumofDisSpent', data=norm_jeju_type_8.nlargest(10, 'NumofDisSpent'), palette='coolwarm')
ax4.set_title('8월 업종별 평균 지원금 이용건수 (NumofDisSpent)')
plt.show()
```
<img src="https://user-images.githubusercontent.com/72846894/103289792-685e6d00-4a2b-11eb-8a08-1ff9340432a8.png"></img>

#### 3) 시간별 소비 상위 10개 업종분석 
```python
def df_time(b,c):
    """
    b는 DisSpent or TotalSpent
    c는 시간을 문자열로
    """
    df_temp= jeju_all.groupby(['Time','Type'])[b].sum()
    df_temp=pd.DataFrame(df_temp).sort_values(b, ascending=False)
    df_temp.reset_index(inplace=True)
    return df_temp[df_temp['Time']==c].reset_index().head(10)
    
df_total_graph=[]
for i in list(('새벽', '오전', '점심', '오후', '저녁', '심야')):
           
    df_temp= jeju_all.groupby(['Time_cut','Type'])['TotalSpent'].sum()
    df_temp=pd.DataFrame(df_temp).sort_values('TotalSpent', ascending=False)
    df_temp.reset_index(inplace=True)
    df_total_graph.append(df_temp[df_temp['Time_cut']==i].reset_index(drop=True).head(15))

    
df_graph = pd.concat(df_total_graph, axis=0)
figure, ((ax1,ax2), (ax3,ax4),(ax5,ax6)) = plt.subplots(nrows=3, ncols=2, constrained_layout=True)
figure.set_size_inches(30,15)

sns.barplot(data=df_graph[df_graph['Time_cut']=='새벽'], x="Type", y="TotalSpent", ax=ax1)  
sns.barplot(data=df_graph[df_graph['Time_cut']=='오전'],  x="Type", y="TotalSpent", ax=ax2)  
sns.barplot(data=df_graph[df_graph['Time_cut']=='점심'],  x="Type", y="TotalSpent", ax=ax3) 
sns.barplot(data=df_graph[df_graph['Time_cut']=='오후'],  x="Type",y="TotalSpent", ax=ax4)  
sns.barplot(data=df_graph[df_graph['Time_cut']=='저녁'],  x="Type", y="TotalSpent", ax=ax5)  
sns.barplot(data=df_graph[df_graph['Time_cut']=='심야'],  x="Type", y="TotalSpent", ax=ax6)

ax1.set_title("새벽", fontsize=20)
ax2.set_title("오전", fontsize=20)
ax3.set_title("점심", fontsize=20)
ax4.set_title("오후", fontsize=20)
ax5.set_title("저녁", fontsize=20)
ax6.set_title("심야", fontsize=20)
plt.suptitle('제주도 시간별 소비현황' ,fontsize=20)
```
<img src="https://user-images.githubusercontent.com/72846894/103289882-9f348300-4a2b-11eb-9687-95ad150bb249.png"></img>
<img src="https://user-images.githubusercontent.com/72846894/103290011-defb6a80-4a2b-11eb-8c15-fdfa1061ddcd.png"></img>


#### 4)업종별 시간대 사용금액(이용건수) 추이분석
- 시간대에 따라 이용건수의 분포가 명확하게 나타날 것으로 생각한 유통/판매 및 외식업종을 기준으로
```python
# 식음료 판매 업종 4가지의 시간대별 총 사용 금액

fig = plt.figure(figsize=(16,12))

ax1 = fig.add_subplot(2, 2, 1)
ax1 = sns.pointplot(x='Time', y='TotalSpent', data=jeju_all[(jeju_all['Type'] == '편의점')].groupby(['Type', 'Time'])['TotalSpent'].sum().reset_index())
ax1.set_title('편의점')
ax2 = fig.add_subplot(2, 2, 2)
ax2 = sns.pointplot(x='Time', y='TotalSpent', data=jeju_all[(jeju_all['Type'] == '슈퍼마켓')].groupby(['Type', 'Time'])['TotalSpent'].sum().reset_index())
ax2.set_title('슈퍼마켓')
ax3 = fig.add_subplot(2, 2, 3)
ax3 = sns.pointplot(x='Time', y='TotalSpent', data=jeju_all[(jeju_all['Type'] == '스넥')].groupby(['Type', 'Time'])['TotalSpent'].sum().reset_index())
ax3.set_title('스넥')
ax4 = fig.add_subplot(2, 2, 4)
ax4 = sns.pointplot(x='Time', y='TotalSpent', data=jeju_all[(jeju_all['Type'] == '기타음료식품')].groupby(['Type', 'Time'])['TotalSpent'].sum().reset_index())
ax4.set_title('기타음료식품')
plt.show()
```
<img src="https://user-images.githubusercontent.com/72846894/103290160-3d284d80-4a2c-11eb-82bc-874a4ab58a9d.png"></img>

#### 5)지역별 소비분석
- 지역별 상위 5개 업종
- 지역별 시간대 사용금액 확인
##### 5-1) 지역별 업종 규모 분포 확인 및 시각화 
```python
df_js2 = jeju_all[jeju_all['FranClass']=='중소2'] #중소2에 대해 추출 
df_js2 = df_js2[['SIGUNGU','FranClass','Type','lon','lat','lon_lat']] #업종별 분포 확인에 대한 정보만 추출 
df_js2.drop_duplicates(inplace=True) #중복되는 row 삭제 
df_js2 #중소2 업종 리스트 

df_js1 = jeju_all[jeju_all['FranClass']=='중소1'] #중소1에 대해 추출 
df_js1 = df_js1[['SIGUNGU','FranClass','Type','lon','lat','lon_lat']] #업종별 분포 확인에 대한 정보만 추출 
df_js1.drop_duplicates(inplace=True) #중복되는 row 삭제 
df_js1 #중소1 업종 리스트 

df_js = jeju_all[jeju_all['FranClass']=='중소'] #중소에 대해 추출 
df_js = df_js[['SIGUNGU','FranClass','Type','lon','lat','lon_lat']] #업종별 분포 확인에 대한 정보만 추출 
df_js.drop_duplicates(inplace=True) #중복되는 row 삭제 
df_js #중소 업종 리스트 

df_gn = jeju_all[jeju_all['FranClass']=='일반'] #일반에 대해 추출 
df_gn = df_gn[['SIGUNGU','FranClass','Type','lon','lat','lon_lat']] #업종별 분포 확인에 대한 정보만 추출 
df_gn.drop_duplicates(inplace=True) #중복되는 row 삭제 
df_gn #일반 업종 리스트 

df_ys = jeju_all[jeju_all['FranClass']=='영세'] #중소에 대해 추출 
df_ys = df_ys[['SIGUNGU','FranClass','Type','lon','lat','lon_lat']] #업종별 분포 확인에 대한 정보만 추출 
df_ys.drop_duplicates(inplace=True) #중복되는 row 삭제 
df_ys #중소 업종 리스트 
```

```python
import pandas as pd 
import folium
from folium.plugins import MarkerCluster, MiniMap

map_jeju = folium.Map((33.38773221915759, 126.54124720118492), zoom_start= 13)
mc = MarkerCluster()

minimap = MiniMap()
map_jeju.add_child(minimap)

map_jeju.add_child(mc)


map = folium.Map((33.38773221915759, 126.54124720118492), zoom_start= 10)

# 중소2 서클 추가 
for i in range(len(df_js2['FranClass'])):
    folium.Circle(list(df_js2.iloc[i][['lat','lon']]), radius=1, color = '#e41a1c',fill_color = '#e41a1c').add_to(map)

# 중소1 서클 추가 
for i in range(len(df_js1['FranClass'])):
    folium.Circle(list(df_js1.iloc[i][['lat','lon']]), radius=1, color = '#377eb8',fill_color = '#d7b5d8').add_to(map)

# 중소 서클 추가 
for i in range(len(df_js['FranClass'])):
    folium.Circle(list(df_js.iloc[i][['lat','lon']]), radius=1, color = '#4daf4a',fill_color = '#df65b0').add_to(map)

# 일반 서클 추가 
for i in range(len(df_gn['FranClass'])):
    folium.Circle(list(df_gn.iloc[i][['lat','lon']]), radius=1, color = '#984ea3',fill_color = '#dd1c77').add_to(map)

# 영세 서클 추가 
for i in range(len(df_ys['FranClass'])):
    folium.Circle(list(df_ys.iloc[i][['lat','lon']]), radius=1, color = '#ff7f00',fill_color = '#980043').add_to(map)    


map
```
<img src="https://user-images.githubusercontent.com/72846894/103291278-b5900e00-4a2e-11eb-88a1-f8cfb2119db9.png"></img>
```python
# 지역별-업종규모별 총사용금액 
f, ax = plt.subplots(figsize=(20,4))
sns.barplot(x='dong_cat', y ='TotalSpent', data=jeju_all, hue='FranClass')
```
<img src="https://user-images.githubusercontent.com/72846894/103290561-3221ed00-4a2d-11eb-993d-00cd7f495aad.png"></img>


### 2. 재난지원금 분석 - 기간별

```python
# 월별 총 사용금액과 월별 재난지원금 총 사용금액 

figure, (ax1,ax2) = plt.subplots(nrows=1, ncols=2)
figure.set_size_inches(18,5)


sns.barplot(data=jeju_all, x="YM", y="TotalSpent", ax=ax1)
sns.barplot(data=jeju_all, x="YM", y="DisSpent", ax=ax2)
```
<img src="https://user-images.githubusercontent.com/72846894/103290735-92b12a00-4a2d-11eb-871e-59a55293f41c.png"></img>
```python
# 전체 기간의 소비 대비 재난지원금 사용비율

fig = plt.figure(figsize=[12, 10])

title = "총 사용금액 중 재난지원금 사용금액 비율"
sizes = [jeju_all.DisSpent.mean()*100/jeju_all.TotalSpent.mean(), 100-(jeju_all.DisSpent.mean()*100/jeju_all.TotalSpent.mean())] 
ax1 = fig.add_subplot(1, 2, 1)
ax1 = plt.pie(sizes, autopct='%1.1f%%', startangle=90)
ax1 = plt.title(title)

title = "총 사용건수 중 재난지원금 사용건수 비율"
sizes = [jeju_all.NumofDisSpent.mean()*100/jeju_all.NumofSpent.mean(), 100-(jeju_all.NumofDisSpent.mean()*100/jeju_all.NumofSpent.mean())] 
ax2 = fig.add_subplot(1, 2, 2)
ax2 = plt.pie(sizes, autopct='%1.1f%%', startangle=90)
ax2 = plt.title(title)

plt.show()
```
<img src="https://user-images.githubusercontent.com/72846894/103290788-ad839e80-4a2d-11eb-842c-0abf0006d4eb.png"></img>
```python
jeju_total = jeju_all.groupby('YM')['TotalSpent'].sum()
jeju_dis = jeju_all.groupby('YM')['DisSpent'].sum()
jeju_month = jeju_total/jeju_all.TotalSpent.sum()
jeju_p = jeju_dis/jeju_total

# 월별 총사용금액 - 총사용금액 대비 재난지원금 확인 

figure, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
figure.set_size_inches(16,4)

sns.barplot(x='YM', y ='TotalSpent', data=jeju_all, color='#b3cde3', ax=ax1)
sns.barplot(x='YM', y ='DisSpent', data=jeju_all, color='#fbb4ae', ax=ax1)


# 총 사용금액의 월별 사용비율 
# 재난지원금 월별 사용 비율
ax2 = jeju_month.plot.bar(x='YM', y='0', rot=0, color='#b3cde3')
for p in ax2.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height())
    x = p.get_x()
    y = p.get_y() + p.get_height()
    ax2.annotate(percentage, (x, y))
ax2 = jeju_p.plot.bar(x='YM', y='0', rot=0, color='#fbb4ae')
for p in ax2.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height())
    x = p.get_x()
    y = p.get_y() + p.get_height()
    ax2.annotate(percentage, (x, y))
```
<img src="https://user-images.githubusercontent.com/72846894/103290846-cc823080-4a2d-11eb-92c0-9dd4753dfefc.png"></img>
```python
sukbak = jeju_all[(jeju_all['Type'].isin(('기타숙박업', '특급호텔', '2급호텔', '1급호텔', '콘도', '렌트카', '기타교통수단')))]

# 숙박 업종의 월별 총 이용건수와 총 이용금액

fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(121)
ax1 = sns.barplot(x='YM', y='NumofSpent', hue='Type', data=sukbak.groupby(['YM', 'Type'], as_index=False)['NumofSpent'].sum(), palette='muted')
ax1.set_title('월별 숙박 이용건수')

ax2 = fig.add_subplot(122)
ax2 = sns.barplot(x='YM', y='TotalSpent', hue='Type', data=sukbak.groupby(['YM', 'Type'], as_index=False)['TotalSpent'].sum(), palette='muted')
ax2.set_title('월별 숙박 이용금액')
plt.show()
```
<img src="https://user-images.githubusercontent.com/72846894/103291058-4ca89600-4a2e-11eb-9336-0ee9515302fc.png"></img>
```python
jeju_all['count'] = 1

# 월별 업종별 총 개수, 총 이용건수와 총 지원금 사용건수로 그룹핑 하기

may_num = jeju_all[jeju_all['YM'] == 202005].groupby(['Type'], as_index=False).agg({'count': sum, 'NumofSpent': sum, 'NumofDisSpent': sum}).sort_values(by='count', ascending=False).head(10)
june_num = jeju_all[jeju_all['YM'] == 202006].groupby(['Type'], as_index=False).agg({'count': sum, 'NumofSpent': sum, 'NumofDisSpent': sum}).sort_values(by='count', ascending=False).head(10)
july_num = jeju_all[jeju_all['YM'] == 202007].groupby(['Type'], as_index=False).agg({'count': sum, 'NumofSpent': sum, 'NumofDisSpent': sum}).sort_values(by='count', ascending=False).head(10)
august_num = jeju_all[jeju_all['YM'] == 202008].groupby(['Type'], as_index=False).agg({'count': sum, 'NumofSpent': sum, 'NumofDisSpent': sum}).sort_values(by='count', ascending=False).head(10)

fig = plt.figure(figsize=(16,12))
ax1 = fig.add_subplot(2, 2, 1)
ax1 = sns.set_color_codes('pastel')
ax1 = sns.barplot(x='NumofSpent', y='Type', data=may_num, label='Total Spent', color='b')
ax1 = sns.set_color_codes('muted')
ax1 = sns.barplot(x='NumofDisSpent', y='Type', data=may_num, label='Fund Spent', color='b')
ax1.legend(loc='best', frameon=True)
ax1.set_title('5월 업종별 TOP10 count의 총이용건수와 총 재난지원금 이용건수')

ax2 = fig.add_subplot(2, 2, 2)
ax2 = sns.set_color_codes('pastel')
ax2 = sns.barplot(x='NumofSpent', y='Type', data=june_num, label='Total Spent', color='b')
ax2 = sns.set_color_codes('muted')
ax2 = sns.barplot(x='NumofDisSpent', y='Type', data=june_num, label='Fund Spent', color='b')
ax2.legend(loc='best', frameon=True)
ax2.set_title('6월 업종별 TOP10 count의 총이용건수와 총 재난지원금 이용건수')

ax3 = fig.add_subplot(2, 2, 3)
ax3 = sns.set_color_codes('pastel')
ax3 = sns.barplot(x='NumofSpent', y='Type', data=july_num, label='Total Spent', color='b')
ax3 = sns.set_color_codes('muted')
ax3 = sns.barplot(x='NumofDisSpent', y='Type', data=july_num, label='Fund Spent', color='b')
ax3.legend(loc='best', frameon=True)
ax3.set_title('7월 업종별 TOP10 count의 총이용건수와 총 재난지원금 이용건수')

ax4 = fig.add_subplot(2, 2, 4)
ax4 = sns.set_color_codes('pastel')
ax4 = sns.barplot(x='NumofSpent', y='Type', data=august_num, label='Total Spent', color='b')
ax4 = sns.set_color_codes('muted')
ax4 = sns.barplot(x='NumofDisSpent', y='Type', data=august_num, label='Fund Spent', color='b')
ax4.legend(loc='best', frameon=True)
ax4.set_title('8월 업종별 TOP10 count의 총이용건수와 총 재난지원금 이용건수')
plt.show()
```
<img src="https://user-images.githubusercontent.com/72846894/103290978-1cf98e00-4a2e-11eb-8595-d7d32846f7e8.png"></img>

```python
# 월별, 업종별로 각각 총 이용건수와 이용금액으로, 총 재난지원금 이용건수와 재난지원금 사용금액으로 그룹핑

type_spent = jeju_all.groupby(['YM', 'Type']).agg({'TotalSpent': sum, 'NumofSpent': sum})
type_dis = jeju_all.groupby(['YM', 'Type']).agg({'DisSpent': sum, 'NumofDisSpent': sum})

fig = plt.figure(figsize=(16,12))
ax1 = fig.add_subplot(2, 2, 1)
ax1 = sns.pointplot(x='YM', y='TotalSpent', hue='Type', data=type_spent['TotalSpent'].groupby('YM', group_keys=False).nlargest(10).reset_index(), palette={'일반한식': 'lightcoral', '슈퍼마켓': 'peru', '편의점': 'goldenrod', '주유소': 'yellowgreen', '면세점': 'mediumseagreen', '서양음식': 'mediumturquoise', '농축협직영매장': 'darkcyan', '유아원': 'dodgerblue', '대형할인점': 'mediumpurple', '골프경기장': 'orchid', '농축수산품': 'palevioletred'})
ax1.set_title('월별 TOP10 업종별 총 이용금액')
ax1.legend(loc='best')

ax2 = fig.add_subplot(2, 2, 2)
ax2 = sns.pointplot(x='YM', y='NumofSpent', hue='Type', data=type_spent['NumofSpent'].groupby('YM', group_keys=False).nlargest(10).reset_index(), palette={'편의점': 'goldenrod', '일반한식': 'lightcoral', '슈퍼마켓': 'peru', '서양음식': 'mediumturquoise', '약국': 'slategrey', '주유소': 'yellowgreen', '농축협직영매장': 'darkcyan', '제과점': 'tan', '스넥': 'dimgray', '면세점': 'mediumseagreen'})
ax2.set_title('월별 TOP10 업종별 총 이용건수')
ax2.legend(loc='best')

ax3 = fig.add_subplot(2, 2, 3)
ax3 = sns.pointplot(x='YM', y='DisSpent', hue='Type', data=type_dis['DisSpent'].groupby('YM', group_keys=False).nlargest(10).reset_index(), palette={'일반한식': 'lightcoral', '슈퍼마켓': 'peru', '농축협직영매장': 'darkcyan', '편의점': 'goldenrod', '주유소': 'yellowgreen', '스포츠레져용품': 'crimson', '농협하나로클럽': 'orangered', '약국': 'slategrey', '정장': 'darkblue', '의원': 'darkgreen', '농축수산품': 'palevioletred', '정육점': 'magenta', '서양음식': 'mediumturquoise'})
ax3.set_title('월별 TOP10 업종별 지원금 이용금액')
ax3.legend(loc='best')

ax4 = fig.add_subplot(2, 2, 4)
ax4 = sns.pointplot(x='YM', y='NumofDisSpent', hue='Type', data=type_dis['NumofDisSpent'].groupby('YM', group_keys=False).nlargest(10).reset_index(), palette={'편의점': 'goldenrod', '슈퍼마켓': 'peru', '일반한식': 'lightcoral', '서양음식': 'mediumturquoise', '농축협직영매장': 'darkcyan', '약국': 'slategrey', '주유소': 'yellowgreen', '제과점': 'tan', '스넥': 'dimgray', '의원': 'darkgreen', '농협하나로클럽': 'orangered', '농축수산품': 'palevioletred'})
ax4.set_title('월별 TOP10 업종별 지원금 이용건수')
ax4.legend(loc='best')
plt.show()
```
<img src="https://user-images.githubusercontent.com/72846894/103291119-68ac3780-4a2e-11eb-9ed0-1af3d932c526.png"></img>

### 3. 재난지원금 분석 - 업종별

```python
# 5월과 8월 데이터를 따로 분리 합니다.

cond1 = jeju_all['YM'] == 202005
cond2 = jeju_all['YM'] == 202008
jeju_may_aug = jeju_all[cond1|cond2]
```

```python
f, ax = plt.subplots(figsize=(20,4))
sns.barplot(x='Category', y ='TotalSpent', data=jeju_may_aug, hue='YM'
```
<img src="https://user-images.githubusercontent.com/72846894/103291725-bf664100-4a2f-11eb-831d-dd86b4d8d0e0.png"></img>

### 4. 재난지원금 분석 - 지역별

```python
f, ax = plt.subplots(figsize=(20,4))
sns.barplot(x='dong_cat', y ='TotalSpent', data=jeju_all, hue='YM')
```
<img src="https://user-images.githubusercontent.com/72846894/103291793-e3298700-4a2f-11eb-9e5a-2f18643cf5c1.png"></img>
```python
f, ax = plt.subplots(figsize=(20,4))
sns.barplot(x='SIGUNGU', y ='TotalSpent', data=jeju_may_aug, hue='YM')
```
<img src="https://user-images.githubusercontent.com/72846894/103291852-02281900-4a30-11eb-9209-4355e9bfb21c.png"></img>

### 5. 재난지원금 - 소상공인 구분 ; 재난지원금 어떤 규모의 소상공인에게 소비 활성화
```python
f, ax = plt.subplots(figsize=(20,4))
sns.barplot(x='FranClass', y ='TotalSpent', data=jeju_all, hue='YM')
```
<img src="https://user-images.githubusercontent.com/72846894/103291977-356aa800-4a30-11eb-9aed-423f6fe10d6d.png"></img>
```python
plt.figure(figsize=(10,10))

jeju_class_norm = jeju_all[jeju_all['FranClass'] == '일반']
df_norm = jeju_class_norm.groupby('Type').sum()
df_norm = df_norm.sort_values(['TotalSpent'], ascending= [False]).head(15)

data = df_norm['TotalSpent'].values[:15]
cat = df_norm['TotalSpent'].index[:15]

plt.pie(data, labels=cat, autopct='%0.1f%%')
plt.legend(cat)

plt.show()
```
<img src="https://user-images.githubusercontent.com/72846894/103292208-b9249480-4a30-11eb-902b-579527949553.png"></img>
```python
figure, ((ax1,ax2), (ax3,ax4)) = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
figure.set_size_inches(18,8)

sns.pointplot(data=jeju_all[jeju_all['FranClass']=='일반'], x="Time", y="TotalSpent", hue='YM',ax=ax1, order = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21', '22', '23', 'x'])
sns.pointplot(data=jeju_all[jeju_all['FranClass']=='중소'], x="Time", y="TotalSpent",hue='YM', ax=ax2, order = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21', '22', '23', 'x'])
sns.pointplot(data=jeju_all[jeju_all['FranClass']=='중소1'], x="Time", y="TotalSpent",hue='YM', ax=ax3, order = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21', '22', '23', 'x'])
sns.pointplot(data=jeju_all[jeju_all['FranClass']=='영세'], x="Time", y="TotalSpent",hue='YM', ax=ax4, order = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21', '22', '23', 'x'])
ax1.set_title("일반")
ax2.set_title("중소")
ax3.set_title("중소1")
ax4.set_title("영세")

plt.show()
```
<img src="https://user-images.githubusercontent.com/72846894/103292278-d6f1f980-4a30-11eb-9d95-3e8a21050729.png"></img>

## 결과

### 1. 제주도의 전반적인 소비분석

#### 1) 월별
- 휴가철인 8월에 소비가 집중될 것으로 예상하였으나., 5~8월 총 사용금액의 차이는 크게 두드러지지 않았습니다. (5월에 풀린 재난지원금의 영향일 것으로 보입니다)
- 전체기간에서 면세점의 사용금액이 압도적으로 우세하였고, 항공사, 대형할인점, 조세서비스, 종합병원 등이 그 뒤를 이었습니다.
#### 2) 시간별
- 총사용금액에 대해 시간별로 많이 소비되는 업종을 살펴보면 새벽에는 주점과 편의점, 오전에는 출근 시간의 영향으로 주유소, 점심시간대와 저녁시간대에는 일반한식 업종이 높게 나타나며 전시간대에 걸쳐 면세점의 소비가 높고, 종합병원, 의원, 골프경기장도 상위권에 속해 있습니다.
- 재난지원금은 재난지원금을 사용할 수 없는 면세점, 대형마트, 유흥주점을 제외하면 총 사용금액과 유사한 소비행태를 보입니다. 또한 전 시간대에 슈퍼마켓, 농협하나로 클럽, 편의점 등 생활밀착형 업종에서 재난지원금이 많이 사용됩니다.
- 제주시와 서귀포시의 총 사용금액을 시간별로 분석하여 상위 10개 업종을 뽑아 봤을 때 두 지역 모두 일반한식, 슈퍼마켓 등 생활 소비업종에서 소비가 높은 것으로 분석해보았습니다. 다만, 제주시는 면세점이 총 소비의 상위권을 차지하고 있으며 주거 밀집지역의 특성상 종합병원과 의원 약국이 높은 것으로 분석됩니다. 서귀포시는 콘도, 골프경기장, 특급호텔에서의 소비가 높은 것을 그래프를 통해 확인할 수 있었습니다.  
#### 3) 업종별
- 각 업종별로 이용건수가 많다고 하더라도 총 사용 금액이 똑같이 높아지지 않습니다.
- 이는 각 업종별로 제공하는 금액대가 다르기 때문에 저렴한 종목에서 여러 건수가 발생한 것이 고가의 종목에서는 건수가 하나만 발생한 것과 같은 이용금액으로 나타날 수 있습니다.
- 이용건수 분포가 명확하게 드러나는 식음료 유통 업종과 외식업종을 보았을 때, 일반적으로 지키는 점심시간(12시)과 저녁시간(18-19시)에 가장 높은 이용 건수와 이용 금액을 볼 수 있었습니다.
- 여행 관련 업종 중 숙박 시설만을 모아서 보았을 때, 대부분 재난지원금 사용이 거의 없는 8월에 가장 많이 소비되었습니다. 이는 8월 관광객 수가 5월 관광객 수의 2배가 넘어 발생하였던 결과인 것으로 분석 되었습니다.
#### 4)지역별
- 지역에 구분없이 일반한식, 슈퍼마켓 등 생활 밀착형 업종에서의 소비금액이  높습니다. 이는 제주도민과 관광객 모두의 영향으로 볼 수 있습니다. 
- 제주시의 경우  면세점의 사용금액이 매우 높으며, 종합병원, 의원, 약국등의 소비가 많은 것으로 보아 생활 밀착 업종이 제주시에 밀집되어 있음을 확인하였습니다.
- 서귀포시의 경우  콘도, 골프경기장 특급호텔에서의 소비가 눈에 띄는 것을 확인하였습니다. 생활권보다 관광지로서의 더욱 발달 되었음을 확인할 수 있습니다.


### 2. 재난지원금 사용 분석 

#### 1)기간별 
- 월별 총사용금액의 경우 차이는 크게 보이지 않으나, 재난지원금의 총 사용금액의 경우 5월에 집중되었음을 확인하였습니다.
- 재난지원금 사용이 집중된 5월, 재난지원금 사용이 낮은 8월의 총 사용금액이 비슷함을 확인하였습니다.
- 5월과 8월의 소비를 분석해보면 어느 지역/업종/업종규모에서의 재난지원금이 사용되었는지 확인할 수 있음을 유추하였고, 이를 기반으로 지역/업종/업종 규모별 분석을 하였습니다. 
#### 2)지역별 
- 지역별 월별 총사용금액을 비교 분석하였을 때, 제주시와 추자면의 경우 5월과 8월의 총 사용금액이 비슷함을 확인하였습니다.
- 제주시의 경우 제주도 핵심 상권 지역으로 대부분의 상점, 편의시설 등이 몰려있으므로 재난지원금 사용이 높았을 것 입니다.
- 추자면의 경우 제주도 북부에 멀리 떨어진 도서 지역입니다. 물리적으로 재난지원금을 다른 지역에서 소비하기 어려웠을 것으로 분석하였습니다. 
#### 3)업종별  
- 의료, 레저/스포츠, 교육/육아, 쇼핑/패션 분야에서 5월 사용금액이 8월 보다 높음을 확인하였고 이는 재난지원금으로 소비가 촉진된 업종으로 분석하였습니다. 
#### 4)업종규모별 
- 일반업종의 경우 재난 지원금 효과가 낮았고 대부분의 업종이 면세점, 골프장, 대형할인장 등의 재난지원금 사용불가 업종에 해당하여 기인한 것으로 분석하였습니다. 




## 함께한 분석가 :thumbsup:
  
- 김경한 
  - 시간별 데이터 분석, 시각화 및 인사이트 도출
  - 데이콘 공유코드 작성 
  - GitHub: https://github.com/darenkim
  
- 김예지
  - 카테고리별 데이터분석, 시각화 및 인사이트 도출
  - ppt 작성
  - GitHub: https://github.com/yeji0701
  
- 이정려
  - 제주도 소비 전반적 시각화
  - Readme 작성
  - GitHub: https://github.com/jungryo
  
- 전예나
  - 위경도 데이터 변환
  - 지역별 데이터분석, 시각화 및 인사이트 도출
  - ppt 작성
  - GitHub: https://github.com/Yenabeam

