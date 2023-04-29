
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import statsmodels.api as sma
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

sns.set_theme()

# Q1
# Loading the data and removing the first 3 rows
data = pd.read_csv("2021-12.csv", skiprows= [1,2,3],index_col=0)

# Removing the nan rows
data.dropna(how='all', inplace=True)

# Setting the date to a date time index
data.index= pd.to_datetime(data.index)


# Q2
# Subsetting the data up until Dec 2019
sub_data= data.loc['01/01/1959':'12/01/2019']


# Q3
# Loading the description table
desc = pd.read_csv("fred_md_desc.csv", skiprows= [1], index_col = 4)

# Removing the nan rows and making sure the columns match
desc.dropna(how='all',inplace=True)

# Filtering only the columns and sorting by ID to match sub_data
desc2 = desc.filter(data.columns, axis=0)
desc3 = desc2.sort_values("id")
print(desc3)

# Q4 & 5 Transforming the series and creating a new data frame

# New dataframe
df = pd.DataFrame()
# 1:1st differences
for i in range(126):
    if (desc3.tcode.iloc[i] == 2.0).any():
        ar = pd.Series((sub_data.iloc[:,i]).diff())
        df[desc3.index[i]] = (ar)

# 2: 2nd differences
for i in range(126):
    if (desc3.tcode.iloc[i] == 3.0).any():
        ar2= pd.Series((sub_data.iloc[:,i]).diff().diff())
        df[desc3.index[i]] = (ar2)

# 3: Log
for i in range(126):
    if (desc3.tcode.iloc[i] == 4.0).any():
        ar3=pd.Series((np.log(sub_data.iloc[:,i])))
        df[desc3.index[i]] = (ar3)

# 4: Log 1st differences
for i in range(126):
    if (desc3.tcode.iloc[i] == 5.0).any():
        ar4=pd.Series((np.log(sub_data.iloc[:,i]).diff()))
        df[desc3.index[i]] = (ar4)

# 5: Log 2nd differences
for i in range(126):
    if (desc3.tcode.iloc[i] == 6.0).any():
        ar5=pd.Series((np.log(sub_data.iloc[:,i]).diff().diff()))
        df[desc3.index[i]] = (ar5)

# 6: Percent Change
for i in range(126):
    if (desc3.tcode.iloc[i]== 7.0).any():
        ar6=pd.Series((sub_data.iloc[:,i]/sub_data.iloc[:,i].shift(1))-1)
        df[desc3.index[i]] = (ar6)

# 0: No transformation
for i in range(126):
    if (desc3.tcode.iloc[i]== 1.0).any():
        ar7=pd.Series(sub_data.iloc[:,i])
        df[desc3.index[i]] = (ar7)

# Independently including the VIXCLSx in the dataframe
df['VIXCLSx'] = sub_data['VIXCLSx']
print(df)

# Re-ordering the columns based on sub_data columns
df = df[sub_data.columns]

# Checking if the columns match
sub_data.columns == df.columns


# Q6 Subtracting mean and dividing standard deviation and filling missing values with 0
stdata = ((df-df.mean())/df.std()).fillna(0)
stdata.to_csv("transformed_data" + '.csv')
stdata


# Q7 Pca function of stdata
def pca_function(stdata):
    """Returns the sign identified 1st principal component of a data set.
    input: stdata - a n x t pandas data frame
    output: 1st principal component, standardised to s.d = 1 and
    signed to have the same sign as the cross sectional mean of the variables"""
    factor1_us = sma.PCA(stdata, 1).factors
    factor1 = (factor1_us - factor1_us.mean()) / factor1_us.std()
    sgn = np.sign(pd.concat([stdata.mean(1), factor1], axis=1).corr().iloc[1, 0])
    return factor1 * sgn
print(pca_function(stdata))


# Q8 Histogram and plot of the pca function stdata
figure, axis = plt.subplots(1, 2, figsize=(14, 6))

axis[0].plot(pca_function(stdata))
axis[0].set_title("Line plot of the factor")

axis[1].hist(pca_function(stdata))
axis[1].set_title("Histogram distribution of the factor")
plt.xlabel('Date')
plt.suptitle('Primary Factor (1st Principle Component)', y=1, size=18)
plt.tight_layout()
figure.savefig("factor"+".pdf")


# Q9 New dataframe of first lags of the transformed data
df_new = df.shift(1)
print(df_new)

# lagging the pca analysis data
factor_lag = pca_function(stdata).shift(1)
print(factor_lag)


# Q10 Regression model of ['INDPRO', 'S&P 500', 'PAYEMS', 'CPIAUCSL', 'BUSINVx'].
regression_data1 = pd.concat([df["INDPRO"], df_new["INDPRO"],factor_lag], axis=1). set_axis (["INDPRO", "ar1", "factor1"],axis=1).dropna()
regression_data2 = pd.concat([df["S&P 500"], df_new["S&P 500"],factor_lag], axis=1). set_axis (["S&P 500", "ar2", "factor2"],axis=1).dropna()
regression_data3 = pd.concat([df["PAYEMS"], df_new["PAYEMS"],factor_lag], axis=1). set_axis (["PAYEMS", "ar3", "factor3"],axis=1).dropna()
regression_data4 = pd.concat([df["CPIAUCSL"], df_new["CPIAUCSL"],factor_lag], axis=1). set_axis (["CPIAUCSL", "ar4", "factor4"],axis=1).dropna()
regression_data5 = pd.concat([df["BUSINVx"], df_new["BUSINVx"],factor_lag], axis=1). set_axis (["BUSINVx", "ar5", "factor5"],axis=1).dropna()

x_vars1 = regression_data1[['ar1', 'factor1']].dropna()
model = sma.OLS(regression_data1["INDPRO"],sma.add_constant(x_vars1))
result = model.fit()
print(result.summary())

x_vars1 = regression_data1[['ar1', 'factor1']].dropna()
model = sma.OLS(regression_data1["INDPRO"],sma.add_constant(x_vars1))
result = model.fit()
print(result.summary())

x_vars2 = regression_data2[['ar2', 'factor2',]].dropna()
model2 = sma.OLS(regression_data2["S&P 500"],sma.add_constant(x_vars2))
result2 = model2.fit()
print(result2.summary())

x_vars3 = regression_data3[['ar3', 'factor3',]].dropna()
model3 = sma.OLS(regression_data3["PAYEMS"],sma.add_constant(x_vars3))
result3 = model3.fit()
print(result3.summary())

x_vars4 = regression_data4[['ar4', 'factor4',]].dropna()
model4 = sma.OLS(regression_data4["CPIAUCSL"],sma.add_constant(x_vars4))
result4 = model4.fit()
print(result4.summary())

x_vars5 = regression_data5[['ar5', 'factor5',]].dropna()
model5 = sma.OLS(regression_data5["BUSINVx"],sma.add_constant(x_vars5))
result5 = model5.fit()
print(result5.summary())


# Q11 A data frame of the fitted values
fitted_values = pd.concat([result.fittedvalues,result2.fittedvalues,result3.fittedvalues,result4.fittedvalues, result5.fittedvalues],axis=1). set_axis (["INDPRO","S&P 500","PAYEMS","CPIAUCSL","BUSINVx"],axis=1)
fitted_values.to_csv("fitted_values" + '.csv')
print(fitted_values)


# Q12 Calling in the nber csv file and setting a date time index and subsetting the data to the appropriate timeframe
nber = pd.read_csv("NBER_DATES.csv",index_col=0)

# Setting the date to a date time index
nber.index = pd.to_datetime(nber.index)

# Subsetting the data up until Dec 2019
nber_data = nber.loc['03/01/1959':'12/01/2019']

# Creating a dataframe of the fitted, actual and nber series
INDPRO = pd.concat ([result.fittedvalues,df["INDPRO"], nber_data],axis=1).set_axis (["INDPRO fitted","Actual","Nber"],axis=1).dropna()
SP = pd.concat ([result2.fittedvalues,df["S&P 500"], nber_data],axis=1).set_axis (["S&P 500 fitted","Actual","Nber"],axis=1).dropna()
PAYEMS = pd.concat ([result3.fittedvalues,df["PAYEMS"], nber_data],axis=1).set_axis (["PAYEMS fitted","Actual","Nber"],axis=1).dropna()
CPIAUCSL = pd.concat ([result4.fittedvalues,df["CPIAUCSL"], nber_data],axis=1).set_axis (["CPIAUCSL fitted","Actual","Nber"],axis=1).dropna()
BUSINVx = pd.concat ([result5.fittedvalues,df["BUSINVx"], nber_data],axis=1).set_axis (["BUSINVx fitted","Actual","Nber"],axis=1).dropna()

# Creating a Seaborn plotfor each of the variables
fig_ind = sns.lmplot(data=INDPRO, x= "INDPRO fitted",y= "Actual", col="Nber")
fig_ind.figure.suptitle('IP Index- First difference of natural log')
fig_ind.figure.tight_layout()
fig_ind.savefig("INDPRO"+".pdf")

fig_SP = sns.lmplot(data=SP, x= "S&P 500 fitted",y= "Actual", col="Nber")
fig_SP.figure.suptitle('S&P s Common Stock Price Index: Composite- First difference of natural log')
fig_SP.figure.tight_layout()
fig_SP.savefig("S&P 500"+".pdf")

fig_PAYEMS = sns.lmplot(data=PAYEMS, x= "PAYEMS fitted",y= "Actual", col="Nber")
fig_PAYEMS.figure.suptitle('All Employees: Total nonfarm - First difference of natural log')
fig_PAYEMS.figure.tight_layout()
fig_PAYEMS.savefig("PAYEMS"+".pdf")

fig_CPIAUCSL = sns.lmplot(data=CPIAUCSL, x= "CPIAUCSL fitted",y= "Actual", col="Nber")
fig_CPIAUCSL.figure.suptitle('CPI : All Items- Second difference of natural log')
fig_CPIAUCSL.figure.tight_layout()
fig_CPIAUCSL.savefig("CPIAUCSL"+".pdf")

fig_BUSINVx = sns.lmplot(data=BUSINVx, x= "BUSINVx fitted",y= "Actual", col="Nber")
fig_BUSINVx.figure.suptitle('Total Business Inventories- First difference of natural log')
fig_BUSINVx.figure.tight_layout()
fig_BUSINVx.savefig("BUSINVx"+".pdf")



