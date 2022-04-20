#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix


# In[2]:


df = pd.read_csv('/Users/rubysun/Desktop/東吳課程_發票資料集/品類資料集/cat_train_v2.csv', header=0)
df.head()


# In[3]:


df.info()


# # Data cleaning

# In[4]:


text_df = df[['product','name']]
text_df.head()


# In[5]:


#shape of data:
text_df.shape


# In[6]:


text_df.info()


# In[7]:


#missing values:
text_df.isnull().sum()


# In[8]:


text_df['product'][3]


# #change data type

# In[9]:


text_df['product'] = text_df['product'].astype(str)


# In[10]:


print(text_df['product'].apply(lambda x: len(x.split(' '))).sum())


# In[11]:


#remove missing value
text_df.dropna(subset=['product'], inplace=True) 


# In[12]:


text_df.isnull().sum()


# In[13]:


#內文長度
text_df['word length'] = text_df['name'].apply(len) 
text_df.head()


# In[14]:


text_df['product'].unique()


# In[15]:


text_df['product'].value_counts()


# In[16]:


# text_df.loc[text_df['product'].isin(['BB霜;粉餅','BB霜;粉底液','BB霜;妝前乳/隔離霜', 'BB霜', 'CC霜', 'CC霜;BB霜',
#                                      'DD霜', 'CC霜;洗面乳', '粉底液', '粉餅', '修容;粉餅', '修容;BB霜;粉底液;CC霜',
#                                      '修容', '修容;眉粉', '眉粉', '眉膠筆', '染眉膏', '眉筆', '蜜粉', '腮紅',
#                                      '蜜粉;BB霜', '妝前乳/隔離霜','遮瑕液', '遮瑕筆', '遮瑕液;遮瑕筆;遮瑕膏',
#                                      '遮瑕液;遮瑕膏', '遮瑕膏', '遮瑕蜜', '遮瑕盤', '蜜粉;遮瑕膏',
#                                      'CC霜;卸妝油']),'product'] = '臉部彩妝'

# text_df.loc[text_df['product'].isin(['卸妝水', '卸妝露','卸妝霜', '卸妝油', '卸妝乳', '卸妝水;卸妝乳',
#                                      '卸妝棉', '卸妝棉;卸妝乳;卸妝油', '卸妝凝膠', '眼唇卸妝','卸甲液',
#                                      '卸妝濕巾']),'product'] = '卸妝產品'

# text_df.loc[text_df['product'].isin(['唇彩;唇釉',  '唇彩;染唇液;唇釉', '唇彩', '唇膏;唇蜜', '唇彩盤', '唇釉', '唇膏',
#                                      '唇膏;唇彩', '唇膏;護手霜;護唇膏', '唇線筆', '唇露', '護唇膏','唇蜜']),'product'] = '唇部彩妝'

# text_df.loc[text_df['product'].isin(['眼影霜', '眼影蜜', '眼影筆', '眼影刷/棒', '眼影打底', '眼線液;眼線膠筆',
#                                      '眼線液筆', '眼線液筆;眼線膠筆', '眼線液筆;眼線膠筆;眼線筆', '眼線筆',
#                                      '眼影筆;護唇膏', '睫毛膏', '眼影盤;腮紅', '眼影盤','眼線膠筆',
#                                      '眼線液','眼霜',]),'product'] = '眼部彩妝'

# text_df.loc[text_df['product'].isin(['維他命B','維他命B;葉黃素;維他命B+C', '維他命B+C', '維他命D', '維他命C',
#                                      '維他命C;維他命B;薑黃','維他命C;維他命E','維他命C;膠原蛋白','維他命E','膠原蛋白;維他命B+C',
#                                      '薑黃', '雞精', '膠原蛋白', '膠原蛋白;維他命B', '葉黃素', '葉黃素;維他命B+C',
#                                      '葉酸', '滴雞精', '維他命B;葉黃素'  ]),'product'] = '保健食品'

# text_df.loc[text_df['product'].isin(['化妝水', '精華油', '面霜/乳霜','面膜', '精華液', '洗面乳;眼霜', '前導精華;眼部精華',
#                                      '前導精華;精華液', '前導精華', '臉部乳液', '護唇精華','臉部乳液;面霜/乳霜','眼部精華']),'product'] = '臉部保養品'


text_df.loc[text_df['product'].isin(['BB霜;粉餅','BB霜;粉底液','BB霜;妝前乳/隔離霜', 'BB霜', 'CC霜', 'CC霜;BB霜',
                                     'DD霜', 'CC霜;洗面乳', '粉底液', '粉餅', '修容;粉餅', '修容;BB霜;粉底液;CC霜',
                                     '修容', '修容;眉粉', '眉粉', '眉膠筆', '染眉膏', '眉筆', '蜜粉', '腮紅',
                                     '蜜粉;BB霜', '妝前乳/隔離霜','遮瑕液', '遮瑕筆', '遮瑕液;遮瑕筆;遮瑕膏',
                                     '遮瑕液;遮瑕膏', '遮瑕膏', '遮瑕蜜', '遮瑕盤', '蜜粉;遮瑕膏',
                                     'CC霜;卸妝油','卸妝水', '卸妝露','卸妝霜', '卸妝油', '卸妝乳', '卸妝水;卸妝乳',
                                     '卸妝棉', '卸妝棉;卸妝乳;卸妝油', '卸妝凝膠', '眼唇卸妝','卸甲液',
                                     '卸妝濕巾','唇彩;唇釉',  '唇彩;染唇液;唇釉', '唇彩', '唇膏;唇蜜', '唇彩盤', '唇釉', '唇膏',
                                     '唇膏;唇彩', '唇膏;護手霜;護唇膏', '唇線筆', '唇露', '護唇膏','唇蜜','眼影霜', '眼影蜜',
                                     '眼影筆', '眼影刷/棒', '眼影打底', '眼線液;眼線膠筆',
                                     '眼線液筆', '眼線液筆;眼線膠筆', '眼線液筆;眼線膠筆;眼線筆', '眼線筆',
                                     '眼影筆;護唇膏', '睫毛膏', '眼影盤;腮紅', '眼影盤','眼線膠筆',
                                     '眼線液','眼霜','化妝水', '精華油', '面霜/乳霜','面膜', '精華液', '洗面乳;眼霜', '前導精華;眼部精華',
                                     '前導精華;精華液', '前導精華', '臉部乳液', '護唇精華','臉部乳液;面霜/乳霜','眼部精華']),'product'] = '彩妝保養'

text_df.loc[text_df['product'].isin([ '中式香腸','火鍋料','即食雞胸','快煮麵/乾拌麵','優格','優酪乳','燕麥奶','鮮乳',
                                     '機能牛乳','保久乳','米漿/豆米漿','羊乳','西式香腸','成人奶粉','即飲奶茶',
                                     '即飲咖啡','即飲烏龍茶;即飲無糖茶','即飲烏龍茶','即飲無糖茶','即飲紅茶',
                                     '咖啡豆','杏仁奶', '豆奶/豆乳','豆漿/黑豆漿','即飲紅茶;即飲無糖茶',
                                     '即飲綠茶','即飲綠茶;即飲無糖茶','即溶咖啡','豆漿/黑豆漿;豆奶/豆乳','麥片穀類',
                                     '調味牛乳','調味牛乳;保久乳','調味豆漿','維他命飲料','原味牛乳','乳酸菌',
                                     '速食麵/泡麵','常溫醬包','RTD調酒','啤酒','氣泡水','益生菌','媽媽茶(哺乳茶)',
                                     '麥片穀類;燕麥奶', '碳酸飲料','礦泉水','椰奶','湯圓','成長幼兒奶粉','媽媽奶粉', '醬油類',
                                     '維他命B','維他命B;葉黃素;維他命B+C', '維他命B+C', '維他命D', '維他命C',
                                     '維他命C;維他命B;薑黃','維他命C;維他命E','維他命C;膠原蛋白','維他命E','膠原蛋白;維他命B+C',
                                     '薑黃', '雞精', '膠原蛋白', '膠原蛋白;維他命B', '葉黃素', '葉黃素;維他命B+C',
                                     '葉酸', '滴雞精', '維他命B;葉黃素']),'product'] = '食品'

text_df.loc[text_df['product'].isin(['洗衣機','吸塵器','冷氣/冷暖空調','快煮壺','果汁機','洗碗機','按摩椅',
                                     '冰箱','吹風機', '電子鍋','電子鍋;電鍋','電風扇','電視遊戲機',
                                     '電暖器','電磁爐','電熱水瓶','電鍋','烘碗機','烘衣機/乾衣機',
                                     '消毒鍋','氣炸鍋','除濕機','烤箱','跑步機','排油煙機','掃地機',
                                     '熱水器','咖啡機','咖啡機;鍋具','液晶電視;電視遊戲機','液晶電視',
                                     '平板電腦', '筆記型電腦', '智慧型手機', '微波爐','溫奶器','調理機',
                                     '快煮壺', '熱水器', '桌上電腦', '智慧錶','滑鼠','鍵盤','淨水器/濾水器']),'product'] = '電器'

text_df.loc[text_df['product'].isin(['人工淚液', '成人牙膏', '水路/健行鞋', '奶瓶', '巧拼地墊', '瓦斯爐(廚房用)',
                                     '瓦斯爐(攜帶式)', '甲片/甲貼', '甲油膠', '安全汽座', '成人牙刷', '兒童牙刷',
                                     '成人紙尿褲', '收納櫃','洗髮精', '沐浴乳', '洗面乳','洗衣精', '洗衣粉',
                                     '沐浴乳;洗髮精(嬰幼童/孕婦)', '身體乳液', '防曬', '兒童牙膏', '兒童漱口水',
                                     '其他地墊(家用)','芳香豆', '保溫杯', '保險套', '狗乾糧罐頭', '狗零食',
                                     '指甲油', '柔軟精', '洗衣皂', '洗衣球', '洗衣精;柔軟精','洗髮精;洗衣球',
                                     '洗髮精(嬰幼童/孕婦)', '洗髮精;面膜', '香氛機','香氛蠟燭', '烤肉架',
                                     '珪藻土地墊','健腹器', '啞鈴/槓鈴', '啞鈴/槓鈴;槓片', '鍋具',
                                     '護手霜', '貓乾糧罐頭', '貓零食', '濕紙巾', '野餐墊',
                                     '棉條', '棉褲/安睡褲','菸', '瑜珈墊', '睡袋', '運動/機能服', '運動鞋',
                                     '槓片', '精油', '精油;香氛機', '廚房紙巾','潤髮乳', '潤髮乳;洗髮精',
                                     '潤髮乳;洗髮精;護髮乳', '潤髮乳;護髮乳', '衛生紙', '衛生棉', '嬰幼手推車',
                                     '嬰幼乳液', '擴香瓶', '濾掛/耳掛咖啡','護髮乳', '護髮乳;洗髮精']),'product'] = '生活用品'


# In[17]:


text_df['product'].value_counts()


# In[18]:


sns.countplot(x='product',data=text_df)
plt.xticks(rotation=45)

plt.xlabel('Type of product')
plt.title('Number of products in each group');
plt.rcParams['font.sans-serif']=['Arial Unicode MS']


# In[19]:


plt.figure(figsize=(8, 5))

text_df[text_df['product'] == '彩妝保養']['word length'].plot(bins=10,kind='hist', color='green', label='彩妝保養', alpha=0.6)
text_df[text_df['product'] == '食品']['word length'].plot(kind='hist', color='blue', label='食品', alpha=0.6)
text_df[text_df['product'] == '電器']['word length'].plot(kind='hist', color='red', label='電器', alpha=0.6)
text_df[text_df['product'] == '生活用品']['word length'].plot(kind='hist', color='yellow', label='生活用品', alpha=0.6)

plt.legend()
plt.xlabel("Length description")
plt.rcParams['font.sans-serif']=['Arial Unicode MS']


# In[20]:


text_df.hist(column='word length', by='product', bins=15,figsize=(12,12));


# In[21]:


import nltk
nltk.download('stopwords')


# In[22]:


def remove_punctuation(name):
    """The function to remove punctuation"""
    table = str.maketrans('', '', string.punctuation)
    return name.translate(table)


# In[23]:


text_df['name'] = text_df['name'].apply(remove_punctuation)
text_df['name'] = text_df['name'].str.replace('[\d]','')
text_df['name'] = text_df['name'].str.replace('[a-zA-z]','')


# In[24]:


text_df.head()


# In[25]:


text_df['name'][1]


# In[26]:


text_df['name'] = text_df['name'].str.replace('[\d]','')


# In[27]:


vectorizer = CountVectorizer()
vectorizer.fit(text_df['name'])
vector = vectorizer.transform(text_df['name'])


# In[28]:


print(vector.shape)
print(vector.toarray())


# In[29]:


# extract the tfid representation matrix of the text data
tfidf_converter = TfidfTransformer()
X_tfidf = tfidf_converter.fit_transform(vector).toarray()
X_tfidf


# In[30]:


X = text_df['name']
y = text_df['product']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)


# In[31]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# # Logistic Regression

# In[32]:


model_log = Pipeline([('vect', CountVectorizer(min_df=5, ngram_range=(1,2))),
                      ('tfidf', TfidfTransformer()),
                      ('model',LogisticRegression()),
                     ])

model_log.fit(X_train, y_train)

ytest = np.array(y_test)
pred = model_log.predict(X_test)


# In[33]:


print('accuracy %s' % accuracy_score(pred, y_test))
print(classification_report(ytest, pred))
print(confusion_matrix(ytest, pred))


# # SVM

# In[34]:


svc = Pipeline([('vect', CountVectorizer(min_df=5, ngram_range=(1,2))),
               ('tfidf', TfidfTransformer()),
               ('model',LinearSVC()),
               ])

svc.fit(X_train, y_train)

ytest = np.array(y_test)
y_pred = svc.predict(X_test)


# In[35]:


print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(ytest, y_pred))
print(confusion_matrix(ytest, y_pred))


# # Naive Bayes Classifier for Multinomial Models

# In[36]:


nbc = Pipeline([('vect', CountVectorizer(min_df=5, ngram_range=(1,2))),
               ('tfidf', TfidfTransformer()),
               ('model',MultinomialNB()),
               ])

nbc.fit(X_train, y_train)

ytest = np.array(y_test)
pred_y = nbc.predict(X_test)


# In[37]:


print('accuracy %s' % accuracy_score(pred_y, y_test))
print(classification_report(ytest, pred_y))
print(confusion_matrix(ytest, pred_y))


# # Random Forest

# In[38]:


rf = Pipeline([('vect', CountVectorizer(min_df=5, ngram_range=(1,2))),
               ('tfidf', TfidfTransformer()),
               ('rf', RandomForestClassifier(n_estimators=50)),
               ])

rf.fit(X_train, y_train)

ytest = np.array(y_test)
preds = rf.predict(X_test)


# In[39]:


print('accuracy %s' % accuracy_score(preds, y_test))
print(classification_report(ytest, preds))
print(confusion_matrix(ytest, preds))


# # Gradient Boosting

# In[40]:


model_gb = Pipeline([('vect', CountVectorizer(min_df=5, ngram_range=(1,2))),
                    ('tfidf', TfidfTransformer()),
                    ('gb', GradientBoostingClassifier(n_estimators=50)),
                    ])

model_gb.fit(X_train, y_train)

ytest = np.array(y_test)
predicted = model_gb.predict(X_test)


# In[41]:



print('accuracy %s' % accuracy_score(predicted, y_test))
print(classification_report(ytest, predicted))
print(confusion_matrix(ytest, predicted))


# ## Model evaluation

# In[42]:


nb_grid = Pipeline([('vect', CountVectorizer(min_df=5, ngram_range=(1,1))),
               ('tfidf', TfidfTransformer(use_idf=False)),
               ('model',MultinomialNB(alpha=0.01)),
               ])

nb_grid.fit(X_train, y_train)

pred_grid = nb_grid.predict(X_test)


# In[43]:


print('accuracy %s' % accuracy_score(pred_grid, y_test))
print(classification_report(ytest, pred_grid))
print(confusion_matrix(ytest, pred_grid))


# In[44]:


log_acc = accuracy_score(pred, y_test)
svm_acc = accuracy_score(y_pred, y_test)
nb_acc = accuracy_score(pred_y, y_test)
rf_acc = accuracy_score(preds, y_test)
gb_acc = accuracy_score(predicted, y_test)
nb_grid_acc = accuracy_score(pred_grid, y_test)


# In[45]:


models = pd.DataFrame({
                      'Model': ['Logistic Regression', 'SVM', 'Naive Bayes', 'Random Forest', 'Gradient Boosting', 'Grid Search_NB'],
                      'Score': [log_acc, svm_acc, nb_acc, rf_acc, gb_acc, nb_grid_acc]})
models.sort_values(by='Score', ascending=False)


# In[ ]:




