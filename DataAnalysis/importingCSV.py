import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

url = "imports-85.data"
path = "carData.csv"
# Assumes data has a header, must set that this data doesn't
df = pd.read_csv(url, header = None)

# print the first 5 rows
print(df.head(5))

# define new headers
headers = ["symboling","normalized-losses","make","fuel-type","aspiration","num-of-doors","body-style","drive-wheels","engine-location","wheel-base","length","width","height","curb-weight","engine-type","num-of-cylinders","engine-size","fuel-system","bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg","price"]

df.columns = headers

# print the first 5 rows
print(df.head(5))

# save modifications
df.to_csv(path)

# check data types
print(df.dtypes)

# get a statistical summary
# only does it for the numerical types
# to include object/string types use 'include = "all"'
print(df.describe(include="all"))
# you can select a column to show its statistics
print(df[['length','compression-ratio']].describe())

# removing missing data
# axis = 0 -> drops the entire row - default
# axis = 1 -> drops the entire column
# inplace = True, does the operation on the given dataframe object
# to replace a value use .replace(missing_value, new_value)
df.replace('?',np.nan,inplace=True)
df.dropna(subset=["price"],inplace=True)


# convert mpg to L/100km
df["city-mpg"] = 235/df["city-mpg"]
# rename the column
df.rename(columns={"city-mpg":"city-L/100km"}, inplace=True)

df["price"] = df["price"].astype("int")


# normalizing data
# .Simple feature scaling
df["length"] = df["length"]/df["length"].max()
# .Min-Max
#df["length"] = (df["length"]-df["length"].min()) / 
#	       (df["length"].max() - df["length"].min() )
# .Z-score
#df["length"] = (df["length"]-df["length"].mean()) / df["length"].std()

# binning
bins = np.linspace(min(df["price"]), max(df["price"]), 4)
group_names = ["Low","Medium","High"]
df["price-binned"] = pd.cut(df["price"], bins, labels=group_names, include_lowest = True)

# turns a categorical variable into a quantitative variable
dumm = pd.get_dummies(df["fuel-type"])
df['gas'] = dumm['gas']
df['diesel'] = dumm['diesel']

y = df["engine-size"]
x = df["price"]
plt.scatter(x,y)
plt.title("Scatterplot of Enfine Size vs Price")
plt.xlabel("Engine Size")
plt.ylabel("Price")
#plt.show()

# grouping categorical variables
df_test = df[['drive-wheels','body-style','price']]
df_grp = df_test.groupby(['drive-wheels','body-style'], as_index=False).mean()

# pivoting
df_pivot = df_grp.pivot(index = 'drive-wheels', columns = 'body-style')
print(df_pivot)

# Heatmap
plt.pcolor(df_pivot, cmap='RdBu')
plt.colorbar()
plt.xlabel("Body Style")
plt.ylabel("Drive Wheels")
#plt.show()

# calculating the correlation
df.dropna(subset=['horsepower'],inplace=True)
df['horsepower'] = df['horsepower'].astype('int')
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print(pearson_coef, p_value)

print(df.corr())
