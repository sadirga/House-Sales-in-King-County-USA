# House-Sales-in-King-County-USA
## Predicting House Prices  
The dataset is from Kaggle [https://www.kaggle.com/harlfoxem/housesalesprediction]   
[![dataset-original.jpg](https://i.postimg.cc/xC836Xfh/dataset-original.jpg)](https://postimg.cc/zbmhBXRF)  

<hr>
  
## Context
This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015.  

## Goals
To predict the house price around King County, which includes Seattle.  

## Features Details:
id - Unique ID for each home sold  
date - Date of the home sale  
price - Price of each home sold  
bedrooms - Number of bedrooms  
bathrooms - Number of bathrooms, where .5 accounts for a room with a toilet but no shower  
sqft_living - Square footage of the interior living space  
sqft_lot - Square footage of the land space  
floors - Number of floors  
waterfront - A dummy variable for whether the apartment was overlooking the waterfront or not  
view - An index from 0 to 4 of how good the view of the property was  
condition - An index from 1 to 5 on the condition of the apartment  
grade - An index from 1 to 13, where 1-3 falls short of building construction and design, 7 has an average level of construction and design, and 11-13 have a high quality level of construction and design.  
sqft_above - The square footage of the interior housing space that is above ground level  
sqft_basement - The square footage of the interior housing space that is below ground level  
yr_built - The year the house was initially built  
yr_renovated - The year of the house’s last renovation  
zipcode - What zipcode area the house is in  
lat - Lattitude  
long - Longitude  
sqft_living15 - The square footage of interior housing living space for the nearest 15 neighbors  
sqft_lot15 - The square footage of the land lots of the nearest 15 neighbors  

<hr>  

As usual, the first thing we need to do is to import the common packages and reading the data
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap, FastMarkerCluster

df = pd.read_csv('kc_house_data.csv', parse_dates=['date'])
pd.options.display.max_columns = 999
#df.info
df.head()
```  
[Output]
	id|date|price|bedrooms|bathrooms|sqft_living|sqft_lot|floors|waterfront|view|condition|grade|sqft_above|sqft_basement|yr_built|yr_renovated|zipcode|lat|long|sqft_living15|sqft_lot15
-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----  
0|7129300520|2014-10-13|221900.0|3|1.00|1180|5650|1.0|0|0|3|7|1180|0|1955|0|98178|47.5112|-122.257|1340|5650
1|6414100192|2014-12-09|538000.0|3|2.25|2570|7242|2.0|0|0|3|7|2170|400|1951|1991|98125|47.7210|-122.319|1690|7639
2|5631500400|2015-02-25|180000.0|2|1.00|770|10000|1.0|0|0|3|6|770|0|1933|0|98028|47.7379|-122.233|2720|8062
3|2487200875|2014-12-09|604000.0|4|3.00|1960|5000|1.0|0|0|5|7|1050|910|1965|0|98136|47.5208|-122.393|1360|5000
4|1954400510|2015-02-18|510000.0|3|2.00|1680|8080|1.0|0|0|3|8|1680|0|1987|0|98074|47.6168|-122.045|1800|75030

Next move is to check if the dataset has missing values or not.

```python
dfDesc = []

for i in df.columns:
    dfDesc.append([
        i,
        df[i].dtypes,
        df[i].isna().sum(),
        ((df[i].isna().sum())/len(df) *100).round(2),
        df[i].nunique(),
        df[i].unique()
    ])
descr = pd.DataFrame(data = dfDesc, columns = ['Features', 'D types', 'Null', 'Null%', 'Unique', 'Unique Value'])
descr    
```  
Features|D types|Null|Null%|Unique|Unique Values
-----|-----|-----|-----|-----|-----|
0|id|int64|0|0.0|21436|[7129300520,|6414100192,|5631500400,|248720087...
1|date|datetime64[ns]|0|0.0|372|[2014-10-13T00:00:00.000000000,|2014-12-09T00:...
2|price|float64|0|0.0|4028|[221900.0,|538000.0,|180000.0,|604000.0,|51000...
3|bedrooms|int64|0|0.0|13|[3,|2,|4,|5,|1,|6,|7,|0,|8,|9,|11,|10,|33]
4|bathrooms|float64|0|0.0|30|[1.0,|2.25,|3.0,|2.0,|4.5,|1.5,|2.5,|1.75,|2.7...
5|sqft_living|int64|0|0.0|1038|[1180,|2570,|770,|1960,|1680,|5420,|1715,|1060...
6|sqft_lot|int64|0|0.0|9782|[5650,|7242,|10000,|5000,|8080,|101930,|6819,|...
7|floors|float64|0|0.0|6|[1.0,|2.0,|1.5,|3.0,|2.5,|3.5]
8|waterfront|int64|0|0.0|2|[0,|1]
9|view|int64|0|0.0|5|[0,|3,|4,|2,|1]
10|condition|int64|0|0.0|5|[3,|5,|4,|1,|2]
11|grade|int64|0|0.0|12|[7,|6,|8,|11,|9,|5,|10,|12,|4,|3,|13,|1]
12|sqft_above|int64|0|0.0|946|[1180,|2170,|770,|1050,|1680,|3890,|1715,|1060...
13|sqft_basement|int64|0|0.0|306|[0,|400,|910,|1530,|730,|1700,|300,|970,|760,|...
14|yr_built|int64|0|0.0|116|[1955,|1951,|1933,|1965,|1987,|2001,|1995,|196...
15|yr_renovated|int64|0|0.0|70|[0,|1991,|2002,|2010,|1999,|1992,|2013,|1994,|...
16|zipcode|int64|0|0.0|70|[98178,|98125,|98028,|98136,|98074,|98053,|980...
17|lat|float64|0|0.0|5034|[47.5112,|47.721000000000004,|47.7379,|47.5208...
18|long|float64|0|0.0|752|[-122.257,|-122.319,|-122.23299999999999,|-122...
19|sqft_living15|int64|0|0.0|777|[1340,|1690,|2720,|1360,|1800,|4760,|2238,|165...
20|sqft_lot15|int64|0|0.0|8689|[5650,|7639,|8062,|5000,|7503,|101930,|6819,|9...

