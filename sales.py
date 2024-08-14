"""
-ID - test kümesi içinde bir (Mağaza, Tarih) ikilisini temsil eden bir Kimlik
-Store - her mağaza için benzersiz bir kimlik
-Sales - belirli bir gündeki ciro (bu sizin tahmin ettiğiniz şeydir)
-Customers - belirli bir gündeki müşteri sayısı
-Open - mağazanın açık olup olmadığını gösteren bir gösterge: 0 = kapalı, 1 = açık
-StateHoliday - resmi tatil anlamına gelir. Normalde birkaç istisna dışında tüm mağazalar resmi tatillerde kapalıdır. Tüm okulların resmi tatillerde ve hafta sonlarında kapalı olduğunu unutmayın. a = resmi tatil, b = Paskalya tatili, c = Noel, 0 = Hiçbiri
-SchoolHoliday - (Store, Date)'in kamu okullarının kapanmasından etkilenip etkilenmediğini gösterir
-StoreType  - 4 farklı mağaza modeli arasında ayrım yapar: a, b, c, d
-Assortment - bir çeşitlilik düzeyini tanımlar: a = temel, b = ekstra, c = genişletilmiş
-CompetitionDistance - en yakın rakip mağazaya olan mesafe (metre cinsinden)
-CompetitionOpenSince[Month/Year] - en yakın rakibin açıldığı yaklaşık yıl ve ayı verir
-Promo - Bir mağazanın o gün promosyon yapıp yapmadığını gösterir
-Promo2 - Promo2, bazı mağazalar için devam eden ve ardışık bir promosyondur: 0 = mağaza katılmıyor, 1 = mağaza katılıyor
-Promo2Since[Year/Week] - Mağazanın Promo2'ye katılmaya başladığı yılı ve takvim haftasını açıklar
-PromoInterval - Promo2'nin başlatıldığı ardışık aralıkları tanımlar ve promosyonun yeniden başlatıldığı ayları adlandırır. Örneğin "Şubat, Mayıs, Ağustos, Kasım" her turun o mağaza için herhangi bir yılın Şubat, Mayıs, Ağustos, Kasım aylarında başladığı anlamına gelir
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
from pandas import to_datetime
from datetime import date
from sklearn.preprocessing import OneHotEncoder

warnings.simplefilter(action="ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 20)
# pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', 20)

################ DATASET YÜKLEME #########
filepath_train = r"train.csv";
filepath_store = r"store.csv";

store = pd.read_csv(filepath_store)
train = pd.read_csv(filepath_train)
train.head()

##########   KÜÇÜK HARF ##########

train.columns = [col.lower() for col in train.columns]
store.columns = [col.lower() for col in store.columns]


########## VERİ BİRLEŞTİRME ###########
########## MAĞAZA AÇIK - SATIŞ VAR ####


train_store = pd.merge(train, store, how = 'inner', on = 'store')
train_store = train_store[(train_store["open"] != 0) & (train_store['sales'] != 0)]


############## VERİ BOYUTU AYARLAMALARI ############

train_store = train_store.sort_values(by = ["store","date"] ,ascending = False)
train_store_filtered = train_store.query("store < 31")

train_store = train_store_filtered

train_store.info()
train_store.head()
train_store.iloc[:]

train_store.shape
#train_store[train_store['store'] == 4].sort_values(by ="date" , ascending = True)
#train_store["date"].min() #min Out[36]: Timestamp('2013-01-01 00:00:00')
#train_store["date"].max() #Out[37]: Timestamp('2015-07-31 00:00:00')

train_store['stateholiday'] = train_store['stateholiday'].replace(0, '0').astype('object')


#####################  VERİ ÖN İŞLEME  ####################

train_store.head(30)
train_store.info()
train_store.tail(30)
train_store.shape
train_store.isnull().sum()

########  Datetime a çevirme   ##########

train_store['date'] = pd.to_datetime(train_store['date'])
train_store.set_index('date', inplace=True)


###### SÜTUN TİP BELİRLEMELERİ ##########


def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(train_store)



##### Değişmesi Gereken Tipler ####

cat_cols = ['promo',"dayofweek","schoolholiday","open", 'promo2', 'promo2sinceyear', 'promo2sinceweek']

for col in cat_cols:
    train_store[col] = train_store[col].astype(object)

print(train_store.dtypes)


cat_cols, num_cols, cat_but_car = grab_col_names(train_store)

train_store.head()
train_store.info()

#### #### #### #### #### #### #### #### #### #### #### #### ####
# groupby incelemesi :

train_store["sales_customers"] = train_store["sales"] / train_store["customers"]

train_store.groupby("assortment").agg({"sales": "mean"})

train_store.groupby("storetype").agg({"sales": "mean"})

train_store.groupby("storetype").agg({"sales_customers": "mean"})
train_store.groupby("assortment").agg({"sales_customers": "mean"})

train_store.groupby("stateholiday").agg({"sales" : "mean"})

train_store["stateholiday"].nunique()
train_store["stateholiday"].value_counts()

train_store[["sales", "promo", "promo2", "promo2sinceweek", "promo2sinceyear", "promointerval"]].groupby(["promo", "promo2", "promo2sinceweek", "promo2sinceyear", "promointerval"]).agg({"sales": "mean"})
pd.set_option('display.max_rows', None)


train.nunique()

train.describe().T



########################################### Kategorik Görselleştirme ##########################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


######## GÖRSELLER İÇ İÇE GELİYOR AYRI AYRI YAPALIM  ????????  ####
######## TARGET GÖRSELLEŞTİRME YAPILMALI KESİNLİKLE ####### ??????


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("######################### *** #########################")

    if plot:
        fig, axs = plt.subplots(2, 4, figsize=(18, 10))
        counts = dataframe[col_name].value_counts()
        ratios = (100 * counts / len(dataframe)).tolist()

        # Count Plot
        plt.subplot(1, 4, 1)
        ax = sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.title("Frequency of " + col_name)
        plt.xticks(rotation=90)

        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., height + 0.01 * max(counts),
                    f'{ratios[i]:.2f}%',
                    ha="center", va="bottom")

        # Pie Chart
        plt.subplot(1, 4, 2)
        values = dataframe[col_name].value_counts()
        plt.pie(x=values, labels=values.index, autopct=lambda p: '{:.2f}% ({:.0f})'.format(p, p / 100 * sum(values)))
        plt.title("Frequency of " + col_name)
        plt.legend(labels=['{} - {:.2f}%'.format(index, value / sum(values) * 100) for index, value in
                           zip(values.index, values)],
                   loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=1)

        # Box Plot (sales)
        plt.subplot(1, 4, 3)
        sns.boxplot(data=dataframe, x=col_name, y="sales", hue="storetype")
        plt.title("sales vs " + col_name)
        plt.xticks(rotation=90)

        # Box Plot (customers)
        plt.subplot(1, 4, 4)
        sns.boxplot(data=dataframe, x=col_name, y="customers", hue="storetype")
        plt.title("customers vs " + col_name)
        plt.xticks(rotation=90)

        plt.tight_layout()
        plt.show()

for col in cat_cols:
    cat_summary(train_store, col, plot=True)

########################################### numerik görselleştirme ##########################################
def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

for col in cat_cols:
    cat_summary(train_store, col)

########################################### describe for numerik ##########################################
def num_summary(df, num_cols):
    quantile=[0.10,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
    print(pd.DataFrame({col: df[num_cols].describe(quantile).T}))

for col in num_cols:
    num_summary(train_store,col)
    train_store[col].hist()
    plt.title(col)
    plt.show()

##########################################  TARGET ANALİZ   ##########################################

# cat değişkenlerin target ile incelemesi
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")
    print("#########################")

for col in cat_cols:
    target_summary_with_cat(train_store,"sales",col)

# num değişkenlerin target ile incelemesi
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(train_store, "sales", col)


##########################################  EKSİK DEĞER   ##########################################
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    """ burda değişkenlerin isimleri işte neymiş n_miss diğeri de ratio bu iki dataframe i neye göre birleştireyim diyor biz de sütunlara göre birleştir diyoruz axis =1 ile """
    print(missing_df, end="\n") # bir boşluk bırakmak için \n koyuyoruz

    if na_name:
        return na_columns

missing_values_table(train_store, True)


##########################################  AYKIRI GÖZLEM   ##########################################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(train_store, col))



# aykırı değere atama yapma :
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(train_store, col)

# atama sonrası kontrol :
for col in num_cols:
    print(col, check_outlier(train_store, col))



##############################################################################################################################

train_store.groupby('storetype')['customers', 'sales'].sum()

# SALES per month trends
import seaborn as sns
import matplotlib.pyplot as plt

sns.catplot(data=train_store, x='NEW_Month', y='sales',
            col='storetype', hue='storetype',
            row='promo', palette='plasma',
            kind='point', height=4, aspect=1.5)

plt.show()

# CUSTOMERS per month trends
# Another example with catplot for customers trends
sns.catplot(data=train_store, x='NEW_Month', y='customers',
            col='storetype', palette='plasma',
            hue='storetype', row='promo', kind='point')

plt.show()

# sale per customer trends
# Plot using catplot instead of factorplot
sns.catplot(data=train_store, x='NEW_Month', y='NEW_Sales_per_Customer',
            col='storetype', palette='plasma',
            hue='storetype', row='promo', kind='point', height=4, aspect=1.5)

plt.show()

# customers
sns.catplot(data=train_store, x='NEW_Month', y='sales',
            col='dayofweek',  # per store type in cols
            palette='plasma',
            hue='storetype',
            row='storetype',  # per store type in rows
            kind='point')  # Kind of plot

plt.show()

# Yukarıdaki grafikler StoreType B'yi en çok satan ve performanslı olarak gösterse de, gerçekte bu doğru değildir.
# En yüksek Müşteri Başına Satış tutarı StoreType D'de gözlemlenir, Promosyonla yaklaşık 12€ ve Promosyon olmadan 10€.
# Mağaza Tipi A ve C'ye gelince, yaklaşık 9€. StoreType B için Düşük Müşteri Başına Satış miktarı,
# Alıcı Sepetini tanımlar: esasen "küçük" şeyler (veya küçük bir miktar) için alışveriş yapan birçok insan vardır.
# Ayrıca, genel olarak bu StoreType'ın dönem boyunca en az miktarda satış ve müşteri oluşturduğunu gördük.


##########################################  FEATURE ENG.   ##########################################
import pandas as pd

# Extract date-based features
train_store['NEW_Day'] = train_store.index.day
train_store['NEW_Month'] = train_store.index.month
train_store['NEW_Year'] = train_store.index.year
train_store['NEW_WeekOfYear'] = train_store.index.isocalendar().week
train_store['NEW_DayOfYear'] = train_store.index.dayofyear
train_store['NEW_IsWeekend'] = train_store.index.weekday >= 5
train_store['NEW_IsMonthStart'] = train_store.index.is_month_start
train_store['NEW_IsMonthEnd'] = train_store.index.is_month_end



# Create lag features for sales
for lag in [1, 7, 30]:
    train_store[f'NEW_Sales_Lag_{lag}'] = train_store.groupby('store')['sales'].shift(lag)

# Create rolling window features for sales
for window in [7, 30]:
    train_store[f'NEW_Rolling_Mean_Sales_{window}'] = train_store.groupby('store')['sales'].transform(lambda x: x.rolling(window).mean())
    train_store[f'NEW_Rolling_Sum_Sales_{window}'] = train_store.groupby('store')['sales'].transform(lambda x: x.rolling(window).sum())
    train_store[f'NEW_Rolling_Std_Sales_{window}'] = train_store.groupby('store')['sales'].transform(lambda x: x.rolling(window).std())

# Create exponential moving average features for sales
for span in [7, 30]:
    train_store[f'NEW_EMA_Sales_{span}'] = train_store.groupby('store')['sales'].transform(lambda x: x.ewm(span=span, adjust=False).mean())

# Create interaction terms
train_store['NEW_Promo_StateHoliday'] = train_store['promo'] * train_store['stateholiday']
train_store['NEW_Promo_SchoolHoliday'] = train_store['promo'] * train_store['schoolholiday']
train_store['NEW_DayOfWeek_Promo'] = train_store['dayofweek'] * train_store['promo']

# Create ratio features
train_store['NEW_Promo2_Ratio'] = train_store['promo2'] / (train_store['competitiondistance'] + 1)

# Encode categorical variables
train_store['NEW_StoreType'] = train_store['storetype'].astype('category').cat.codes
train_store['NEW_Assortment'] = train_store['assortment'].astype('category').cat.codes

# Display the first few rows of the modified train_store dataframe
print(train_store.head())

#### #### #### #### #### #### #### FEATURELARI KATEGORİZE #### #### #### #### #### #### ####

cat_cols, num_cols, cat_but_car = grab_col_names(train_store)



#### #### #### #### #### #### #### FEATURE SONRASI MISSING VALUES #### #### #### #### #### #### ####

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    """ burda değişkenlerin isimleri işte neymiş n_miss diğeri de ratio bu iki dataframe i neye göre birleştireyim diyor biz de sütunlara göre birleştir diyoruz axis =1 ile """
    print(missing_df, end="\n") # bir boşluk bırakmak için \n koyuyoruz

    if na_name:
        return na_columns

missing_values_table(train_store, True)


########################### Random Noise #########################################

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


## Lag/Shifted Features
def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

train_store = lag_features(train_store, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])


### Rolling Mean Features

def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe


train_store = roll_mean_features(train_store, [365, 546])

### Exponentially Weighted Mean Features

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

train_store = ewm_features(train_store, alphas, lags)


########################################################################################################################
# HATA DÜZELTME :
########################################################################################################################
# Sütun isimlerini temizleme fonksiyonu
def clean_column_names(df):
    df.columns = df.columns.str.replace('[^A-Za-z0-9]+', '_', regex=True)
    return df

# Sütun isimlerini temizle
train_store= clean_column_names(train_store)
# Sütun isimlerini temizleme ve UTF-8 kodlamasıyla ilgili hataları düzeltme fonksiyonu

def clean_column_names(df):
    df.columns = [col.encode('utf-8', 'ignore').decode('utf-8').replace('[^A-Za-z0-9]+', '_') for col in df.columns]
    df.columns = df.columns.str.replace('[^A-Za-z0-9]+', '_', regex=True)
    return df

# Sütun isimlerini temizle
train_store = clean_column_names(train_store)
########################################################################################################################


## One-Hot Encoding


train_store['stateholiday'] = train_store['stateholiday'].replace(0, '0').astype('object')

train_store = pd.get_dummies(train_store, columns=["stateholiday","storetype","assortment","promointerval",'store',
                                                   'dayofweek',"open","promo","schoolholiday","promo2","promo2sinceyear",
                                                   "promo2sinceweek", "NEW_Promo_StateHoliday","NEW_Promo_SchoolHoliday","NEW_Promo2_Ratio",
                                                   "NEW_DayOfWeek_Promo"])

########################################################################################################################

## Converting sales to log(1+sales)

train_store['sales'] = np.log1p(train_store["sales"].values)

####  Model


def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False



###########################Time-Based Validation Sets ###################################

import pandas as pd

total_data = 22900
#Yeni oranlara göre veri bölme
train_size = int(0.80 * total_data)
val_size = int(0.10 * total_data)
test_size = total_data - train_size - val_size

#Veriyi bölme
train_data = train_store.iloc[:train_size]
val_data = train_store.iloc[train_size:train_size + val_size]
test_data = train_store.iloc[train_size + val_size:]

#Boyutları kontrol etme
print(f"Train set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")
print(f"Test set size: {len(test_data)}")


cols = [col for col in train_data.columns if col not in ["sales"]]

Y_train = train_data['sales']
X_train = train_data[cols]

Y_val = val_data['sales']
X_val = val_data[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape

########################################################################################################################
# LightGBM ile Zaman Serisi Modeli
########################################################################################################################

# !pip install lightgbm
# conda install lightgbm

import lightgbm as lgb
# LightGBM parameters
lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 5000,
              'early_stopping_rounds': 1000,
              'nthread': -1}


lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)


model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  feval=lgbm_smape)

#### LOGDAN ÇIKARMA :
train_store['sales'] = np.expm1(train_store['sales'].values)


################### SONUÇ HESAPLAMA : ############################
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Tahmin yapma
y_pred_log = model.predict(X_val)

# Logaritmik dönüşümden çıkartma
y_pred = np.expm1(y_pred_log)
y_val_exp = np.expm1(Y_val)

# Hata metriklerini hesaplama
mae = mean_absolute_error(y_val_exp, y_pred)

# sMAPE hesaplama
def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

smape_score = smape(y_val_exp, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"sMAPE: {smape_score}")

############## Değişken Önem Düzeyleri  ##############

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, num=200)

plot_lgb_importances(model, num=30, plot=True)


feat_imp = plot_lgb_importances(model, num=200)

importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values

imp_feats = [col for col in cols if col not in importance_zero]
len(imp_feats)

########################################################################################################################
# İlk 20 tahmini ekrana yazdırma
print("\nİlk 20 Tahmin:")
for i in range(min(20, len(y_pred))):
    print(f"Index: {i}, Tahmin: {y_pred[i]}, Gerçek Değer: {y_val_exp.iloc[i]}, Özellikler: {X_val.iloc[i].to_dict()}")


# Tahmin edilen değerleri X_val veri setine yeni bir sütun olarak ekleme
X_val['y_pred'] = y_pred

# İlk birkaç satırı kontrol edelim
print(X_val.head())

########################################################################################################################

### predictleri ve sales'i tek bir df'e atma

tahmin_df = val_data
tahmin_df["Y_pred"] = y_pred
tahmin_df["sales"] = np.expm1(tahmin_df["sales"])
tahmin_df_6 = tahmin_df[tahmin_df["store_6"]==True]


X_val[X_val["store_6"]==True]["y_pred"]
X_val[X_val["store_6"]==True]["y_val"]


# Tahmin ve gerçek değerlerin grafiğini çizme (sadece seçilen store için)
plt.figure(figsize=(10, 5))
plt.plot(tahmin_df_6["Y_pred"]
, label='Gerçek Değerler')
plt.plot(tahmin_df_6["sales"]
, label='Tahminler', alpha=0.7)
plt.xlabel('Örnekler')
plt.ylabel('Değerler')
plt.legend()
plt.show()
