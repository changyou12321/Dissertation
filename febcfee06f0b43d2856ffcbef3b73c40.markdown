---
jupyter:
  kernelspec:
    display_name: Python 3.8.1 64-bit
    language: python
    name: python38164bitec02e5d543514226ba2d0c1724834648
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.8.1
  nbformat: 4
  nbformat_minor: 2
---

::: {.cell .code execution_count="1"}
``` python
#!/usr/bin/env python3
#！_*_ coding:utf-8 _*_

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import shap
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from pylab import mpl
import seaborn as sns 
import random 
import os 
import gc
from tqdm import tqdm 
from sklearn.preprocessing import label_binarize

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False   

import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(123)

print('import finish')
```

::: {.output .stream .stdout}
    import finish
:::
:::

::: {.cell .markdown}
### 1、读取数据 {#1读取数据}
:::

::: {.cell .code execution_count="2"}
``` python
# 文件路径 

DATA_PATH_ONS = '../data/ons-model-based-income-estimates-msoa (4).xls'
DATA_PATH_HOUSING = '../data/housing-density-borough (2).csv'
DATA_PATH_LOCAL = '../data/local_authority_traffic (5).csv'

# 读取数据 
data_ons_2011_12 = pd.read_excel(DATA_PATH_ONS,sheet_name='2011-12 (weekly income)')
data_ons_2013_14 = pd.read_excel(DATA_PATH_ONS,sheet_name='2013-14 (weekly income)')
data_ons_2015_16 = pd.read_excel(DATA_PATH_ONS,sheet_name='2015-16 (annual income)')

data_housing = pd.read_csv(DATA_PATH_HOUSING)
data_local = pd.read_csv(DATA_PATH_LOCAL)
```
:::

::: {.cell .code execution_count="3" scrolled="true"}
``` python
data_ons_2011_12
```

::: {.output .execute_result execution_count="3"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSOA code</th>
      <th>MSOA name</th>
      <th>Local authority code</th>
      <th>Local authority name</th>
      <th>Region code</th>
      <th>Region name</th>
      <th>Total weekly income (£)</th>
      <th>Upper confidence limit (£)</th>
      <th>Lower confidence limit (£)</th>
      <th>Confidence interval (£)</th>
      <th>...</th>
      <th>Lower confidence limit (£).1</th>
      <th>Confidence interval (£).1</th>
      <th>Net income before housing costs (£)</th>
      <th>Upper confidence limit (£).2</th>
      <th>Lower confidence limit (£).2</th>
      <th>Confidence interval (£).2</th>
      <th>Net income after housing costs (£)</th>
      <th>Upper confidence limit (£).3</th>
      <th>Lower confidence limit (£).3</th>
      <th>Confidence interval (£).3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E02004297</td>
      <td>County Durham 001</td>
      <td>E06000047</td>
      <td>County Durham</td>
      <td>E12000001</td>
      <td>North East</td>
      <td>630</td>
      <td>690</td>
      <td>570</td>
      <td>120</td>
      <td>...</td>
      <td>440</td>
      <td>80</td>
      <td>480</td>
      <td>530</td>
      <td>440</td>
      <td>90</td>
      <td>450</td>
      <td>510</td>
      <td>390</td>
      <td>120</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E02004290</td>
      <td>County Durham 002</td>
      <td>E06000047</td>
      <td>County Durham</td>
      <td>E12000001</td>
      <td>North East</td>
      <td>730</td>
      <td>800</td>
      <td>660</td>
      <td>140</td>
      <td>...</td>
      <td>500</td>
      <td>90</td>
      <td>510</td>
      <td>560</td>
      <td>460</td>
      <td>100</td>
      <td>460</td>
      <td>530</td>
      <td>400</td>
      <td>120</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E02004298</td>
      <td>County Durham 003</td>
      <td>E06000047</td>
      <td>County Durham</td>
      <td>E12000001</td>
      <td>North East</td>
      <td>690</td>
      <td>760</td>
      <td>630</td>
      <td>130</td>
      <td>...</td>
      <td>480</td>
      <td>90</td>
      <td>500</td>
      <td>550</td>
      <td>450</td>
      <td>100</td>
      <td>470</td>
      <td>540</td>
      <td>420</td>
      <td>130</td>
    </tr>
    <tr>
      <th>3</th>
      <td>E02004299</td>
      <td>County Durham 004</td>
      <td>E06000047</td>
      <td>County Durham</td>
      <td>E12000001</td>
      <td>North East</td>
      <td>540</td>
      <td>600</td>
      <td>500</td>
      <td>100</td>
      <td>...</td>
      <td>390</td>
      <td>70</td>
      <td>420</td>
      <td>470</td>
      <td>390</td>
      <td>80</td>
      <td>380</td>
      <td>440</td>
      <td>340</td>
      <td>100</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E02004291</td>
      <td>County Durham 005</td>
      <td>E06000047</td>
      <td>County Durham</td>
      <td>E12000001</td>
      <td>North East</td>
      <td>500</td>
      <td>550</td>
      <td>460</td>
      <td>90</td>
      <td>...</td>
      <td>370</td>
      <td>70</td>
      <td>420</td>
      <td>460</td>
      <td>380</td>
      <td>80</td>
      <td>370</td>
      <td>420</td>
      <td>320</td>
      <td>100</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7196</th>
      <td>W02000362</td>
      <td>Newport 016</td>
      <td>W06000022</td>
      <td>Newport</td>
      <td>W92000004</td>
      <td>Wales</td>
      <td>810</td>
      <td>900</td>
      <td>730</td>
      <td>180</td>
      <td>...</td>
      <td>550</td>
      <td>120</td>
      <td>570</td>
      <td>630</td>
      <td>510</td>
      <td>120</td>
      <td>520</td>
      <td>600</td>
      <td>450</td>
      <td>150</td>
    </tr>
    <tr>
      <th>7197</th>
      <td>W02000363</td>
      <td>Newport 017</td>
      <td>W06000022</td>
      <td>Newport</td>
      <td>W92000004</td>
      <td>Wales</td>
      <td>540</td>
      <td>600</td>
      <td>480</td>
      <td>120</td>
      <td>...</td>
      <td>390</td>
      <td>80</td>
      <td>440</td>
      <td>490</td>
      <td>400</td>
      <td>90</td>
      <td>400</td>
      <td>460</td>
      <td>350</td>
      <td>110</td>
    </tr>
    <tr>
      <th>7198</th>
      <td>W02000364</td>
      <td>Newport 018</td>
      <td>W06000022</td>
      <td>Newport</td>
      <td>W92000004</td>
      <td>Wales</td>
      <td>440</td>
      <td>480</td>
      <td>390</td>
      <td>90</td>
      <td>...</td>
      <td>330</td>
      <td>70</td>
      <td>340</td>
      <td>380</td>
      <td>310</td>
      <td>70</td>
      <td>250</td>
      <td>290</td>
      <td>220</td>
      <td>70</td>
    </tr>
    <tr>
      <th>7199</th>
      <td>W02000365</td>
      <td>Newport 019</td>
      <td>W06000022</td>
      <td>Newport</td>
      <td>W92000004</td>
      <td>Wales</td>
      <td>550</td>
      <td>610</td>
      <td>500</td>
      <td>110</td>
      <td>...</td>
      <td>400</td>
      <td>80</td>
      <td>410</td>
      <td>460</td>
      <td>370</td>
      <td>80</td>
      <td>340</td>
      <td>390</td>
      <td>300</td>
      <td>100</td>
    </tr>
    <tr>
      <th>7200</th>
      <td>W02000366</td>
      <td>Newport 020</td>
      <td>W06000022</td>
      <td>Newport</td>
      <td>W92000004</td>
      <td>Wales</td>
      <td>910</td>
      <td>1030</td>
      <td>810</td>
      <td>210</td>
      <td>...</td>
      <td>590</td>
      <td>130</td>
      <td>570</td>
      <td>630</td>
      <td>510</td>
      <td>120</td>
      <td>490</td>
      <td>570</td>
      <td>430</td>
      <td>150</td>
    </tr>
  </tbody>
</table>
<p>7201 rows × 22 columns</p>
</div>
```
:::
:::

::: {.cell .code execution_count="4"}
``` python
data_ons_2013_14
```

::: {.output .execute_result execution_count="4"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSOA code</th>
      <th>MSOA name</th>
      <th>Local authority code</th>
      <th>Local authority name</th>
      <th>Region code</th>
      <th>Region name</th>
      <th>Total weekly income (£)</th>
      <th>Upper confidence limit (£)</th>
      <th>Lower confidence limit (£)</th>
      <th>Confidence interval (£)</th>
      <th>...</th>
      <th>Lower confidence limit (£).1</th>
      <th>Confidence interval (£).1</th>
      <th>Net income before housing costs (£)</th>
      <th>Upper confidence limit (£).2</th>
      <th>Lower confidence limit (£).2</th>
      <th>Confidence interval (£).2</th>
      <th>Net income after housing costs (£)</th>
      <th>Upper confidence limit (£).3</th>
      <th>Lower confidence limit (£).3</th>
      <th>Confidence interval (£).3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E02004297</td>
      <td>County Durham 001</td>
      <td>E06000047</td>
      <td>County Durham</td>
      <td>E12000001</td>
      <td>North East</td>
      <td>700</td>
      <td>790</td>
      <td>610</td>
      <td>180</td>
      <td>...</td>
      <td>480</td>
      <td>90</td>
      <td>510</td>
      <td>550</td>
      <td>470</td>
      <td>80</td>
      <td>480</td>
      <td>550</td>
      <td>430</td>
      <td>120</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E02004290</td>
      <td>County Durham 002</td>
      <td>E06000047</td>
      <td>County Durham</td>
      <td>E12000001</td>
      <td>North East</td>
      <td>730</td>
      <td>830</td>
      <td>640</td>
      <td>190</td>
      <td>...</td>
      <td>520</td>
      <td>100</td>
      <td>500</td>
      <td>540</td>
      <td>460</td>
      <td>80</td>
      <td>480</td>
      <td>540</td>
      <td>420</td>
      <td>120</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E02004298</td>
      <td>County Durham 003</td>
      <td>E06000047</td>
      <td>County Durham</td>
      <td>E12000001</td>
      <td>North East</td>
      <td>730</td>
      <td>830</td>
      <td>640</td>
      <td>190</td>
      <td>...</td>
      <td>510</td>
      <td>100</td>
      <td>550</td>
      <td>600</td>
      <td>510</td>
      <td>90</td>
      <td>500</td>
      <td>570</td>
      <td>440</td>
      <td>130</td>
    </tr>
    <tr>
      <th>3</th>
      <td>E02004299</td>
      <td>County Durham 004</td>
      <td>E06000047</td>
      <td>County Durham</td>
      <td>E12000001</td>
      <td>North East</td>
      <td>600</td>
      <td>690</td>
      <td>530</td>
      <td>160</td>
      <td>...</td>
      <td>440</td>
      <td>80</td>
      <td>450</td>
      <td>490</td>
      <td>420</td>
      <td>70</td>
      <td>420</td>
      <td>470</td>
      <td>370</td>
      <td>100</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E02004291</td>
      <td>County Durham 005</td>
      <td>E06000047</td>
      <td>County Durham</td>
      <td>E12000001</td>
      <td>North East</td>
      <td>540</td>
      <td>620</td>
      <td>470</td>
      <td>140</td>
      <td>...</td>
      <td>370</td>
      <td>70</td>
      <td>430</td>
      <td>460</td>
      <td>390</td>
      <td>70</td>
      <td>390</td>
      <td>440</td>
      <td>340</td>
      <td>100</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7196</th>
      <td>W02000362</td>
      <td>Newport 016</td>
      <td>W06000022</td>
      <td>Newport</td>
      <td>W92000004</td>
      <td>Wales</td>
      <td>720</td>
      <td>820</td>
      <td>630</td>
      <td>190</td>
      <td>...</td>
      <td>510</td>
      <td>100</td>
      <td>510</td>
      <td>560</td>
      <td>470</td>
      <td>90</td>
      <td>470</td>
      <td>540</td>
      <td>420</td>
      <td>120</td>
    </tr>
    <tr>
      <th>7197</th>
      <td>W02000363</td>
      <td>Newport 017</td>
      <td>W06000022</td>
      <td>Newport</td>
      <td>W92000004</td>
      <td>Wales</td>
      <td>570</td>
      <td>650</td>
      <td>500</td>
      <td>150</td>
      <td>...</td>
      <td>410</td>
      <td>80</td>
      <td>420</td>
      <td>460</td>
      <td>390</td>
      <td>70</td>
      <td>400</td>
      <td>460</td>
      <td>360</td>
      <td>100</td>
    </tr>
    <tr>
      <th>7198</th>
      <td>W02000364</td>
      <td>Newport 018</td>
      <td>W06000022</td>
      <td>Newport</td>
      <td>W92000004</td>
      <td>Wales</td>
      <td>460</td>
      <td>520</td>
      <td>400</td>
      <td>130</td>
      <td>...</td>
      <td>330</td>
      <td>80</td>
      <td>330</td>
      <td>360</td>
      <td>300</td>
      <td>60</td>
      <td>260</td>
      <td>300</td>
      <td>230</td>
      <td>70</td>
    </tr>
    <tr>
      <th>7199</th>
      <td>W02000365</td>
      <td>Newport 019</td>
      <td>W06000022</td>
      <td>Newport</td>
      <td>W92000004</td>
      <td>Wales</td>
      <td>540</td>
      <td>610</td>
      <td>470</td>
      <td>140</td>
      <td>...</td>
      <td>390</td>
      <td>70</td>
      <td>390</td>
      <td>420</td>
      <td>360</td>
      <td>60</td>
      <td>340</td>
      <td>390</td>
      <td>300</td>
      <td>90</td>
    </tr>
    <tr>
      <th>7200</th>
      <td>W02000366</td>
      <td>Newport 020</td>
      <td>W06000022</td>
      <td>Newport</td>
      <td>W92000004</td>
      <td>Wales</td>
      <td>910</td>
      <td>1030</td>
      <td>790</td>
      <td>240</td>
      <td>...</td>
      <td>620</td>
      <td>160</td>
      <td>610</td>
      <td>680</td>
      <td>550</td>
      <td>120</td>
      <td>560</td>
      <td>640</td>
      <td>490</td>
      <td>140</td>
    </tr>
  </tbody>
</table>
<p>7201 rows × 22 columns</p>
</div>
```
:::
:::

::: {.cell .code execution_count="5" scrolled="true"}
``` python
data_ons_2015_16
```

::: {.output .execute_result execution_count="5"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSOA code</th>
      <th>MSOA name</th>
      <th>Local authority code</th>
      <th>Local authority name</th>
      <th>Region code</th>
      <th>Region name</th>
      <th>Total annual income (£)</th>
      <th>Upper confidence limit (£)</th>
      <th>Lower confidence limit (£)</th>
      <th>Confidence interval (£)</th>
      <th>...</th>
      <th>Lower confidence limit (£).1</th>
      <th>Confidence interval (£).1</th>
      <th>Net income before housing costs (£)</th>
      <th>Upper confidence limit (£).2</th>
      <th>Lower confidence limit (£).2</th>
      <th>Confidence interval (£).2</th>
      <th>Net income after housing costs (£)</th>
      <th>Upper confidence limit (£).3</th>
      <th>Lower confidence limit (£).3</th>
      <th>Confidence interval (£).3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E02004297</td>
      <td>County Durham 001</td>
      <td>E06000047</td>
      <td>County Durham</td>
      <td>E12000001</td>
      <td>North East</td>
      <td>35900</td>
      <td>45200</td>
      <td>28500</td>
      <td>16700</td>
      <td>...</td>
      <td>22100</td>
      <td>11700</td>
      <td>27600</td>
      <td>33300</td>
      <td>22800</td>
      <td>10400</td>
      <td>25600</td>
      <td>31000</td>
      <td>21200</td>
      <td>9800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E02004290</td>
      <td>County Durham 002</td>
      <td>E06000047</td>
      <td>County Durham</td>
      <td>E12000001</td>
      <td>North East</td>
      <td>42500</td>
      <td>53600</td>
      <td>33700</td>
      <td>19900</td>
      <td>...</td>
      <td>23900</td>
      <td>13100</td>
      <td>28600</td>
      <td>34500</td>
      <td>23700</td>
      <td>10800</td>
      <td>27500</td>
      <td>33200</td>
      <td>22700</td>
      <td>10500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E02004298</td>
      <td>County Durham 003</td>
      <td>E06000047</td>
      <td>County Durham</td>
      <td>E12000001</td>
      <td>North East</td>
      <td>38000</td>
      <td>47700</td>
      <td>30200</td>
      <td>17600</td>
      <td>...</td>
      <td>22800</td>
      <td>12300</td>
      <td>28200</td>
      <td>34100</td>
      <td>23400</td>
      <td>10700</td>
      <td>26700</td>
      <td>32300</td>
      <td>22100</td>
      <td>10200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>E02004299</td>
      <td>County Durham 004</td>
      <td>E06000047</td>
      <td>County Durham</td>
      <td>E12000001</td>
      <td>North East</td>
      <td>33500</td>
      <td>42200</td>
      <td>26700</td>
      <td>15500</td>
      <td>...</td>
      <td>21600</td>
      <td>11200</td>
      <td>25500</td>
      <td>30800</td>
      <td>21100</td>
      <td>9700</td>
      <td>22400</td>
      <td>27100</td>
      <td>18500</td>
      <td>8700</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E02004291</td>
      <td>County Durham 005</td>
      <td>E06000047</td>
      <td>County Durham</td>
      <td>E12000001</td>
      <td>North East</td>
      <td>31700</td>
      <td>39800</td>
      <td>25200</td>
      <td>14600</td>
      <td>...</td>
      <td>20700</td>
      <td>10800</td>
      <td>25100</td>
      <td>30200</td>
      <td>20800</td>
      <td>9500</td>
      <td>20900</td>
      <td>25300</td>
      <td>17200</td>
      <td>8000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7196</th>
      <td>W02000362</td>
      <td>Newport 016</td>
      <td>W06000022</td>
      <td>Newport</td>
      <td>W92000004</td>
      <td>Wales</td>
      <td>38800</td>
      <td>48800</td>
      <td>30900</td>
      <td>17900</td>
      <td>...</td>
      <td>24300</td>
      <td>12600</td>
      <td>27700</td>
      <td>33400</td>
      <td>23000</td>
      <td>10500</td>
      <td>26000</td>
      <td>31400</td>
      <td>21500</td>
      <td>9800</td>
    </tr>
    <tr>
      <th>7197</th>
      <td>W02000363</td>
      <td>Newport 017</td>
      <td>W06000022</td>
      <td>Newport</td>
      <td>W92000004</td>
      <td>Wales</td>
      <td>32700</td>
      <td>41200</td>
      <td>26000</td>
      <td>15100</td>
      <td>...</td>
      <td>20200</td>
      <td>10400</td>
      <td>25000</td>
      <td>30200</td>
      <td>20800</td>
      <td>9400</td>
      <td>22700</td>
      <td>27400</td>
      <td>18800</td>
      <td>8600</td>
    </tr>
    <tr>
      <th>7198</th>
      <td>W02000364</td>
      <td>Newport 018</td>
      <td>W06000022</td>
      <td>Newport</td>
      <td>W92000004</td>
      <td>Wales</td>
      <td>25900</td>
      <td>32800</td>
      <td>20300</td>
      <td>12500</td>
      <td>...</td>
      <td>17500</td>
      <td>9300</td>
      <td>20200</td>
      <td>24400</td>
      <td>16600</td>
      <td>7800</td>
      <td>16400</td>
      <td>19800</td>
      <td>13600</td>
      <td>6300</td>
    </tr>
    <tr>
      <th>7199</th>
      <td>W02000365</td>
      <td>Newport 019</td>
      <td>W06000022</td>
      <td>Newport</td>
      <td>W92000004</td>
      <td>Wales</td>
      <td>32800</td>
      <td>41400</td>
      <td>26000</td>
      <td>15400</td>
      <td>...</td>
      <td>20400</td>
      <td>10700</td>
      <td>24200</td>
      <td>29200</td>
      <td>20000</td>
      <td>9200</td>
      <td>19300</td>
      <td>23400</td>
      <td>16000</td>
      <td>7400</td>
    </tr>
    <tr>
      <th>7200</th>
      <td>W02000366</td>
      <td>Newport 020</td>
      <td>W06000022</td>
      <td>Newport</td>
      <td>W92000004</td>
      <td>Wales</td>
      <td>50300</td>
      <td>63700</td>
      <td>39800</td>
      <td>23900</td>
      <td>...</td>
      <td>28600</td>
      <td>15000</td>
      <td>30200</td>
      <td>36600</td>
      <td>25000</td>
      <td>11600</td>
      <td>28200</td>
      <td>34100</td>
      <td>23300</td>
      <td>10800</td>
    </tr>
  </tbody>
</table>
<p>7201 rows × 22 columns</p>
</div>
```
:::
:::

::: {.cell .code execution_count="6"}
``` python
# 数据拼接 

data_2011_2016 = pd.concat([data_ons_2011_12,data_ons_2013_14,data_ons_2015_16],axis = 0)
data_2011_2016
```

::: {.output .execute_result execution_count="6"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSOA code</th>
      <th>MSOA name</th>
      <th>Local authority code</th>
      <th>Local authority name</th>
      <th>Region code</th>
      <th>Region name</th>
      <th>Total weekly income (£)</th>
      <th>Upper confidence limit (£)</th>
      <th>Lower confidence limit (£)</th>
      <th>Confidence interval (£)</th>
      <th>...</th>
      <th>Net income before housing costs (£)</th>
      <th>Upper confidence limit (£).2</th>
      <th>Lower confidence limit (£).2</th>
      <th>Confidence interval (£).2</th>
      <th>Net income after housing costs (£)</th>
      <th>Upper confidence limit (£).3</th>
      <th>Lower confidence limit (£).3</th>
      <th>Confidence interval (£).3</th>
      <th>Total annual income (£)</th>
      <th>Net annual income (£)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E02004297</td>
      <td>County Durham 001</td>
      <td>E06000047</td>
      <td>County Durham</td>
      <td>E12000001</td>
      <td>North East</td>
      <td>630.0</td>
      <td>690</td>
      <td>570</td>
      <td>120</td>
      <td>...</td>
      <td>480</td>
      <td>530</td>
      <td>440</td>
      <td>90</td>
      <td>450</td>
      <td>510</td>
      <td>390</td>
      <td>120</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E02004290</td>
      <td>County Durham 002</td>
      <td>E06000047</td>
      <td>County Durham</td>
      <td>E12000001</td>
      <td>North East</td>
      <td>730.0</td>
      <td>800</td>
      <td>660</td>
      <td>140</td>
      <td>...</td>
      <td>510</td>
      <td>560</td>
      <td>460</td>
      <td>100</td>
      <td>460</td>
      <td>530</td>
      <td>400</td>
      <td>120</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E02004298</td>
      <td>County Durham 003</td>
      <td>E06000047</td>
      <td>County Durham</td>
      <td>E12000001</td>
      <td>North East</td>
      <td>690.0</td>
      <td>760</td>
      <td>630</td>
      <td>130</td>
      <td>...</td>
      <td>500</td>
      <td>550</td>
      <td>450</td>
      <td>100</td>
      <td>470</td>
      <td>540</td>
      <td>420</td>
      <td>130</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>E02004299</td>
      <td>County Durham 004</td>
      <td>E06000047</td>
      <td>County Durham</td>
      <td>E12000001</td>
      <td>North East</td>
      <td>540.0</td>
      <td>600</td>
      <td>500</td>
      <td>100</td>
      <td>...</td>
      <td>420</td>
      <td>470</td>
      <td>390</td>
      <td>80</td>
      <td>380</td>
      <td>440</td>
      <td>340</td>
      <td>100</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E02004291</td>
      <td>County Durham 005</td>
      <td>E06000047</td>
      <td>County Durham</td>
      <td>E12000001</td>
      <td>North East</td>
      <td>500.0</td>
      <td>550</td>
      <td>460</td>
      <td>90</td>
      <td>...</td>
      <td>420</td>
      <td>460</td>
      <td>380</td>
      <td>80</td>
      <td>370</td>
      <td>420</td>
      <td>320</td>
      <td>100</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7196</th>
      <td>W02000362</td>
      <td>Newport 016</td>
      <td>W06000022</td>
      <td>Newport</td>
      <td>W92000004</td>
      <td>Wales</td>
      <td>NaN</td>
      <td>48800</td>
      <td>30900</td>
      <td>17900</td>
      <td>...</td>
      <td>27700</td>
      <td>33400</td>
      <td>23000</td>
      <td>10500</td>
      <td>26000</td>
      <td>31400</td>
      <td>21500</td>
      <td>9800</td>
      <td>38800.0</td>
      <td>29900.0</td>
    </tr>
    <tr>
      <th>7197</th>
      <td>W02000363</td>
      <td>Newport 017</td>
      <td>W06000022</td>
      <td>Newport</td>
      <td>W92000004</td>
      <td>Wales</td>
      <td>NaN</td>
      <td>41200</td>
      <td>26000</td>
      <td>15100</td>
      <td>...</td>
      <td>25000</td>
      <td>30200</td>
      <td>20800</td>
      <td>9400</td>
      <td>22700</td>
      <td>27400</td>
      <td>18800</td>
      <td>8600</td>
      <td>32700.0</td>
      <td>24900.0</td>
    </tr>
    <tr>
      <th>7198</th>
      <td>W02000364</td>
      <td>Newport 018</td>
      <td>W06000022</td>
      <td>Newport</td>
      <td>W92000004</td>
      <td>Wales</td>
      <td>NaN</td>
      <td>32800</td>
      <td>20300</td>
      <td>12500</td>
      <td>...</td>
      <td>20200</td>
      <td>24400</td>
      <td>16600</td>
      <td>7800</td>
      <td>16400</td>
      <td>19800</td>
      <td>13600</td>
      <td>6300</td>
      <td>25900.0</td>
      <td>21700.0</td>
    </tr>
    <tr>
      <th>7199</th>
      <td>W02000365</td>
      <td>Newport 019</td>
      <td>W06000022</td>
      <td>Newport</td>
      <td>W92000004</td>
      <td>Wales</td>
      <td>NaN</td>
      <td>41400</td>
      <td>26000</td>
      <td>15400</td>
      <td>...</td>
      <td>24200</td>
      <td>29200</td>
      <td>20000</td>
      <td>9200</td>
      <td>19300</td>
      <td>23400</td>
      <td>16000</td>
      <td>7400</td>
      <td>32800.0</td>
      <td>25200.0</td>
    </tr>
    <tr>
      <th>7200</th>
      <td>W02000366</td>
      <td>Newport 020</td>
      <td>W06000022</td>
      <td>Newport</td>
      <td>W92000004</td>
      <td>Wales</td>
      <td>NaN</td>
      <td>63700</td>
      <td>39800</td>
      <td>23900</td>
      <td>...</td>
      <td>30200</td>
      <td>36600</td>
      <td>25000</td>
      <td>11600</td>
      <td>28200</td>
      <td>34100</td>
      <td>23300</td>
      <td>10800</td>
      <td>50300.0</td>
      <td>35300.0</td>
    </tr>
  </tbody>
</table>
<p>21603 rows × 24 columns</p>
</div>
```
:::
:::

::: {.cell .code execution_count="7" scrolled="true"}
``` python
data_housing
```

::: {.output .execute_result execution_count="7"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Code</th>
      <th>Name</th>
      <th>Year</th>
      <th>Source</th>
      <th>Population</th>
      <th>Inland_Area _Hectares</th>
      <th>Total_Area_Hectares</th>
      <th>Population_per_hectare</th>
      <th>Square_Kilometres</th>
      <th>Population_per_square_kilometre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E09000001</td>
      <td>City of London</td>
      <td>1999</td>
      <td>ONS MYE</td>
      <td>6581</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>22.7</td>
      <td>2.9</td>
      <td>2266.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E09000001</td>
      <td>City of London</td>
      <td>2000</td>
      <td>ONS MYE</td>
      <td>7014</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>24.2</td>
      <td>2.9</td>
      <td>2415.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E09000001</td>
      <td>City of London</td>
      <td>2001</td>
      <td>ONS MYE</td>
      <td>7359</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>25.3</td>
      <td>2.9</td>
      <td>2534.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>E09000001</td>
      <td>City of London</td>
      <td>2002</td>
      <td>ONS MYE</td>
      <td>7280</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>25.1</td>
      <td>2.9</td>
      <td>2506.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E09000001</td>
      <td>City of London</td>
      <td>2003</td>
      <td>ONS MYE</td>
      <td>7115</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>24.5</td>
      <td>2.9</td>
      <td>2450.1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1867</th>
      <td>E13000002</td>
      <td>Outer London</td>
      <td>2046</td>
      <td>GLA Population Projections</td>
      <td>6573194</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>52.4</td>
      <td>1254.2</td>
      <td>5240.8</td>
    </tr>
    <tr>
      <th>1868</th>
      <td>E13000002</td>
      <td>Outer London</td>
      <td>2047</td>
      <td>GLA Population Projections</td>
      <td>6598789</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>52.6</td>
      <td>1254.2</td>
      <td>5261.2</td>
    </tr>
    <tr>
      <th>1869</th>
      <td>E13000002</td>
      <td>Outer London</td>
      <td>2048</td>
      <td>GLA Population Projections</td>
      <td>6622921</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>52.8</td>
      <td>1254.2</td>
      <td>5280.4</td>
    </tr>
    <tr>
      <th>1870</th>
      <td>E13000002</td>
      <td>Outer London</td>
      <td>2049</td>
      <td>GLA Population Projections</td>
      <td>6647527</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>53.0</td>
      <td>1254.2</td>
      <td>5300.1</td>
    </tr>
    <tr>
      <th>1871</th>
      <td>E13000002</td>
      <td>Outer London</td>
      <td>2050</td>
      <td>GLA Population Projections</td>
      <td>6671295</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>53.2</td>
      <td>1254.2</td>
      <td>5319.0</td>
    </tr>
  </tbody>
</table>
<p>1872 rows × 10 columns</p>
</div>
```
:::
:::

::: {.cell .code execution_count="8" scrolled="true"}
``` python
data_local
```

::: {.output .execute_result execution_count="8"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>local_authority_name</th>
      <th>ons_code</th>
      <th>region_id</th>
      <th>local_authority_id</th>
      <th>year</th>
      <th>link_length_km</th>
      <th>link_length_miles</th>
      <th>cars_and_taxis</th>
      <th>all_motor_vehicles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>321</td>
      <td>Hartlepool</td>
      <td>E06000001</td>
      <td>11</td>
      <td>132</td>
      <td>1993</td>
      <td>385.351</td>
      <td>239.45</td>
      <td>287107346.8</td>
      <td>3.422533e+08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>822</td>
      <td>Hartlepool</td>
      <td>E06000001</td>
      <td>11</td>
      <td>132</td>
      <td>1993</td>
      <td>385.351</td>
      <td>239.45</td>
      <td>287107346.8</td>
      <td>3.422533e+08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>823</td>
      <td>Middlesbrough</td>
      <td>E06000002</td>
      <td>11</td>
      <td>170</td>
      <td>1993</td>
      <td>493.212</td>
      <td>306.47</td>
      <td>604140895.9</td>
      <td>7.048122e+08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>824</td>
      <td>Redcar and Cleveland</td>
      <td>E06000003</td>
      <td>11</td>
      <td>171</td>
      <td>1993</td>
      <td>675.168</td>
      <td>419.53</td>
      <td>446328948.9</td>
      <td>5.228484e+08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>825</td>
      <td>Stockton-on-Tees</td>
      <td>E06000004</td>
      <td>11</td>
      <td>163</td>
      <td>1993</td>
      <td>845.721</td>
      <td>525.51</td>
      <td>668148077.7</td>
      <td>8.000379e+08</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5730</th>
      <td>6551</td>
      <td>Torfaen</td>
      <td>W06000020</td>
      <td>4</td>
      <td>25</td>
      <td>2020</td>
      <td>482.089</td>
      <td>299.56</td>
      <td>284507214.7</td>
      <td>3.601668e+08</td>
    </tr>
    <tr>
      <th>5731</th>
      <td>6552</td>
      <td>Monmouthshire</td>
      <td>W06000021</td>
      <td>4</td>
      <td>13</td>
      <td>2020</td>
      <td>1626.520</td>
      <td>1010.67</td>
      <td>529506425.6</td>
      <td>7.323918e+08</td>
    </tr>
    <tr>
      <th>5732</th>
      <td>6553</td>
      <td>Newport</td>
      <td>W06000022</td>
      <td>4</td>
      <td>6</td>
      <td>2020</td>
      <td>757.706</td>
      <td>470.82</td>
      <td>743499944.2</td>
      <td>1.000086e+09</td>
    </tr>
    <tr>
      <th>5733</th>
      <td>6554</td>
      <td>Powys</td>
      <td>W06000023</td>
      <td>4</td>
      <td>14</td>
      <td>2020</td>
      <td>5357.025</td>
      <td>3328.70</td>
      <td>586677064.9</td>
      <td>8.497405e+08</td>
    </tr>
    <tr>
      <th>5734</th>
      <td>6555</td>
      <td>Merthyr Tydfil</td>
      <td>W06000024</td>
      <td>4</td>
      <td>21</td>
      <td>2020</td>
      <td>347.960</td>
      <td>216.21</td>
      <td>160076300.7</td>
      <td>2.043523e+08</td>
    </tr>
  </tbody>
</table>
<p>5735 rows × 10 columns</p>
</div>
```
:::
:::

::: {.cell .code execution_count="9"}
``` python
# cars_and_taxis，
# Population，
# Population_per_square_kilometre，
# Total_Area_Hectares，
# Total weekly income (£)， 
# Net income before housing costs (£)， 
# Net income after housing costs (£)
```
:::

::: {.cell .code execution_count="10"}
``` python
data_housing['cars_and_taxis'] = data_local['cars_and_taxis'] 
data_housing['Total weekly income (£)'] = data_ons_2011_12['Total weekly income (£)']
data_housing['Net income before housing costs (£)'] = data_ons_2011_12['Net income before housing costs (£)']
data_housing['Net income after housing costs (£)'] = data_ons_2011_12['Net income after housing costs (£)']

# 标签 
data_housing['all_motor_vehicles'] = data_local['all_motor_vehicles']
```
:::

::: {.cell .code execution_count="11"}
``` python
data_local.isnull().sum()
```

::: {.output .execute_result execution_count="11"}
    id                      0
    local_authority_name    0
    ons_code                0
    region_id               0
    local_authority_id      0
    year                    0
    link_length_km          0
    link_length_miles       0
    cars_and_taxis          0
    all_motor_vehicles      0
    dtype: int64
:::
:::

::: {.cell .code execution_count="12"}
``` python
data_housing
```

::: {.output .execute_result execution_count="12"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Code</th>
      <th>Name</th>
      <th>Year</th>
      <th>Source</th>
      <th>Population</th>
      <th>Inland_Area _Hectares</th>
      <th>Total_Area_Hectares</th>
      <th>Population_per_hectare</th>
      <th>Square_Kilometres</th>
      <th>Population_per_square_kilometre</th>
      <th>cars_and_taxis</th>
      <th>Total weekly income (£)</th>
      <th>Net income before housing costs (£)</th>
      <th>Net income after housing costs (£)</th>
      <th>all_motor_vehicles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E09000001</td>
      <td>City of London</td>
      <td>1999</td>
      <td>ONS MYE</td>
      <td>6581</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>22.7</td>
      <td>2.9</td>
      <td>2266.2</td>
      <td>287107346.8</td>
      <td>630</td>
      <td>480</td>
      <td>450</td>
      <td>3.422533e+08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E09000001</td>
      <td>City of London</td>
      <td>2000</td>
      <td>ONS MYE</td>
      <td>7014</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>24.2</td>
      <td>2.9</td>
      <td>2415.3</td>
      <td>287107346.8</td>
      <td>730</td>
      <td>510</td>
      <td>460</td>
      <td>3.422533e+08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E09000001</td>
      <td>City of London</td>
      <td>2001</td>
      <td>ONS MYE</td>
      <td>7359</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>25.3</td>
      <td>2.9</td>
      <td>2534.1</td>
      <td>604140895.9</td>
      <td>690</td>
      <td>500</td>
      <td>470</td>
      <td>7.048122e+08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>E09000001</td>
      <td>City of London</td>
      <td>2002</td>
      <td>ONS MYE</td>
      <td>7280</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>25.1</td>
      <td>2.9</td>
      <td>2506.9</td>
      <td>446328948.9</td>
      <td>540</td>
      <td>420</td>
      <td>380</td>
      <td>5.228484e+08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E09000001</td>
      <td>City of London</td>
      <td>2003</td>
      <td>ONS MYE</td>
      <td>7115</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>24.5</td>
      <td>2.9</td>
      <td>2450.1</td>
      <td>668148077.7</td>
      <td>500</td>
      <td>420</td>
      <td>370</td>
      <td>8.000379e+08</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1867</th>
      <td>E13000002</td>
      <td>Outer London</td>
      <td>2046</td>
      <td>GLA Population Projections</td>
      <td>6573194</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>52.4</td>
      <td>1254.2</td>
      <td>5240.8</td>
      <td>856437997.9</td>
      <td>630</td>
      <td>470</td>
      <td>410</td>
      <td>1.055627e+09</td>
    </tr>
    <tr>
      <th>1868</th>
      <td>E13000002</td>
      <td>Outer London</td>
      <td>2047</td>
      <td>GLA Population Projections</td>
      <td>6598789</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>52.6</td>
      <td>1254.2</td>
      <td>5261.2</td>
      <td>425162959.2</td>
      <td>710</td>
      <td>490</td>
      <td>460</td>
      <td>5.072792e+08</td>
    </tr>
    <tr>
      <th>1869</th>
      <td>E13000002</td>
      <td>Outer London</td>
      <td>2048</td>
      <td>GLA Population Projections</td>
      <td>6622921</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>52.8</td>
      <td>1254.2</td>
      <td>5280.4</td>
      <td>356124062.4</td>
      <td>540</td>
      <td>400</td>
      <td>360</td>
      <td>4.169537e+08</td>
    </tr>
    <tr>
      <th>1870</th>
      <td>E13000002</td>
      <td>Outer London</td>
      <td>2049</td>
      <td>GLA Population Projections</td>
      <td>6647527</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>53.0</td>
      <td>1254.2</td>
      <td>5300.1</td>
      <td>784174777.9</td>
      <td>450</td>
      <td>390</td>
      <td>320</td>
      <td>1.001045e+09</td>
    </tr>
    <tr>
      <th>1871</th>
      <td>E13000002</td>
      <td>Outer London</td>
      <td>2050</td>
      <td>GLA Population Projections</td>
      <td>6671295</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>53.2</td>
      <td>1254.2</td>
      <td>5319.0</td>
      <td>659300533.2</td>
      <td>490</td>
      <td>360</td>
      <td>310</td>
      <td>8.107151e+08</td>
    </tr>
  </tbody>
</table>
<p>1872 rows × 15 columns</p>
</div>
```
:::
:::

::: {.cell .code execution_count="13"}
``` python
# 拷贝一份

df = data_housing.copy()
df_rf = data_housing.copy()
df_mlp = data_housing.copy()
df 
```

::: {.output .execute_result execution_count="13"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Code</th>
      <th>Name</th>
      <th>Year</th>
      <th>Source</th>
      <th>Population</th>
      <th>Inland_Area _Hectares</th>
      <th>Total_Area_Hectares</th>
      <th>Population_per_hectare</th>
      <th>Square_Kilometres</th>
      <th>Population_per_square_kilometre</th>
      <th>cars_and_taxis</th>
      <th>Total weekly income (£)</th>
      <th>Net income before housing costs (£)</th>
      <th>Net income after housing costs (£)</th>
      <th>all_motor_vehicles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E09000001</td>
      <td>City of London</td>
      <td>1999</td>
      <td>ONS MYE</td>
      <td>6581</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>22.7</td>
      <td>2.9</td>
      <td>2266.2</td>
      <td>287107346.8</td>
      <td>630</td>
      <td>480</td>
      <td>450</td>
      <td>3.422533e+08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E09000001</td>
      <td>City of London</td>
      <td>2000</td>
      <td>ONS MYE</td>
      <td>7014</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>24.2</td>
      <td>2.9</td>
      <td>2415.3</td>
      <td>287107346.8</td>
      <td>730</td>
      <td>510</td>
      <td>460</td>
      <td>3.422533e+08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E09000001</td>
      <td>City of London</td>
      <td>2001</td>
      <td>ONS MYE</td>
      <td>7359</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>25.3</td>
      <td>2.9</td>
      <td>2534.1</td>
      <td>604140895.9</td>
      <td>690</td>
      <td>500</td>
      <td>470</td>
      <td>7.048122e+08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>E09000001</td>
      <td>City of London</td>
      <td>2002</td>
      <td>ONS MYE</td>
      <td>7280</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>25.1</td>
      <td>2.9</td>
      <td>2506.9</td>
      <td>446328948.9</td>
      <td>540</td>
      <td>420</td>
      <td>380</td>
      <td>5.228484e+08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E09000001</td>
      <td>City of London</td>
      <td>2003</td>
      <td>ONS MYE</td>
      <td>7115</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>24.5</td>
      <td>2.9</td>
      <td>2450.1</td>
      <td>668148077.7</td>
      <td>500</td>
      <td>420</td>
      <td>370</td>
      <td>8.000379e+08</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1867</th>
      <td>E13000002</td>
      <td>Outer London</td>
      <td>2046</td>
      <td>GLA Population Projections</td>
      <td>6573194</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>52.4</td>
      <td>1254.2</td>
      <td>5240.8</td>
      <td>856437997.9</td>
      <td>630</td>
      <td>470</td>
      <td>410</td>
      <td>1.055627e+09</td>
    </tr>
    <tr>
      <th>1868</th>
      <td>E13000002</td>
      <td>Outer London</td>
      <td>2047</td>
      <td>GLA Population Projections</td>
      <td>6598789</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>52.6</td>
      <td>1254.2</td>
      <td>5261.2</td>
      <td>425162959.2</td>
      <td>710</td>
      <td>490</td>
      <td>460</td>
      <td>5.072792e+08</td>
    </tr>
    <tr>
      <th>1869</th>
      <td>E13000002</td>
      <td>Outer London</td>
      <td>2048</td>
      <td>GLA Population Projections</td>
      <td>6622921</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>52.8</td>
      <td>1254.2</td>
      <td>5280.4</td>
      <td>356124062.4</td>
      <td>540</td>
      <td>400</td>
      <td>360</td>
      <td>4.169537e+08</td>
    </tr>
    <tr>
      <th>1870</th>
      <td>E13000002</td>
      <td>Outer London</td>
      <td>2049</td>
      <td>GLA Population Projections</td>
      <td>6647527</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>53.0</td>
      <td>1254.2</td>
      <td>5300.1</td>
      <td>784174777.9</td>
      <td>450</td>
      <td>390</td>
      <td>320</td>
      <td>1.001045e+09</td>
    </tr>
    <tr>
      <th>1871</th>
      <td>E13000002</td>
      <td>Outer London</td>
      <td>2050</td>
      <td>GLA Population Projections</td>
      <td>6671295</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>53.2</td>
      <td>1254.2</td>
      <td>5319.0</td>
      <td>659300533.2</td>
      <td>490</td>
      <td>360</td>
      <td>310</td>
      <td>8.107151e+08</td>
    </tr>
  </tbody>
</table>
<p>1872 rows × 15 columns</p>
</div>
```
:::
:::

::: {.cell .markdown}
### 2、绘图 {#2绘图}
:::

::: {.cell .code execution_count="14"}
``` python
# soource plot 

plt.figure(figsize=(5,6))
plt.title('Soure count plot')
sns.countplot(df['Source'])
plt.show()
```

::: {.output .display_data}
![](vertopal_febcfee06f0b43d2856ffcbef3b73c40/c87d797d8b944f559483f776094adb1b2a30e890.png)
:::
:::

::: {.cell .code execution_count="15"}
``` python
print('各字段缺失值占比：')
print(df.isnull().sum().sort_values(ascending=False) / df.shape[0])
```

::: {.output .stream .stdout}
    各字段缺失值占比：
    Code                                   0.0
    Name                                   0.0
    Year                                   0.0
    Source                                 0.0
    Population                             0.0
    Inland_Area _Hectares                  0.0
    Total_Area_Hectares                    0.0
    Population_per_hectare                 0.0
    Square_Kilometres                      0.0
    Population_per_square_kilometre        0.0
    cars_and_taxis                         0.0
    Total weekly income (£)                0.0
    Net income before housing costs (£)    0.0
    Net income after housing costs (£)     0.0
    all_motor_vehicles                     0.0
    dtype: float64
:::
:::

::: {.cell .code execution_count="16" scrolled="true"}
``` python
# 设置字体 

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

cols = [col for col in df.columns if col not in ['Code','Year','Name','Source']]


# 每个字段的核密度曲线图 
for col in cols:
    print(col)
    plt.figure(figsize=(7,5))
    sns.kdeplot(df[col])
    plt.show()
```

::: {.output .stream .stdout}
    Population
:::

::: {.output .display_data}
![](vertopal_febcfee06f0b43d2856ffcbef3b73c40/a63df0267a95663eabf91b73938d0c3ee32eb433.png)
:::

::: {.output .stream .stdout}
    Inland_Area _Hectares
:::

::: {.output .display_data}
![](vertopal_febcfee06f0b43d2856ffcbef3b73c40/d528816d3cf361ae32ffbe618e460d992e9da257.png)
:::

::: {.output .stream .stdout}
    Total_Area_Hectares
:::

::: {.output .display_data}
![](vertopal_febcfee06f0b43d2856ffcbef3b73c40/ddbf1ee7f4881ae59802dd63e650208465cfd438.png)
:::

::: {.output .stream .stdout}
    Population_per_hectare
:::

::: {.output .display_data}
![](vertopal_febcfee06f0b43d2856ffcbef3b73c40/71be78218c99bf4ca11a0e01e8e745e9e303407b.png)
:::

::: {.output .stream .stdout}
    Square_Kilometres
:::

::: {.output .display_data}
![](vertopal_febcfee06f0b43d2856ffcbef3b73c40/8bc6905f5555200757bfee2d0060620903ddd808.png)
:::

::: {.output .stream .stdout}
    Population_per_square_kilometre
:::

::: {.output .display_data}
![](vertopal_febcfee06f0b43d2856ffcbef3b73c40/31113c7b101603ef0a0797c73c212ae1accfd628.png)
:::

::: {.output .stream .stdout}
    cars_and_taxis
:::

::: {.output .display_data}
![](vertopal_febcfee06f0b43d2856ffcbef3b73c40/8f005b429b52a722afe851c7fbfe8b07c379fe92.png)
:::

::: {.output .stream .stdout}
    Total weekly income (£)
:::

::: {.output .display_data}
![](vertopal_febcfee06f0b43d2856ffcbef3b73c40/b391c2a861be1d2eea294d9d12b1cc2a5e3468ac.png)
:::

::: {.output .stream .stdout}
    Net income before housing costs (£)
:::

::: {.output .display_data}
![](vertopal_febcfee06f0b43d2856ffcbef3b73c40/5872e9a404233cc934b00481ddf499004e61abe1.png)
:::

::: {.output .stream .stdout}
    Net income after housing costs (£)
:::

::: {.output .display_data}
![](vertopal_febcfee06f0b43d2856ffcbef3b73c40/8be5f885473b7ea56789644e323f8fb67db58fa7.png)
:::

::: {.output .stream .stdout}
    all_motor_vehicles
:::

::: {.output .display_data}
![](vertopal_febcfee06f0b43d2856ffcbef3b73c40/424947b9bfc8224d894c99440955cc06fc544fc8.png)
:::
:::

::: {.cell .code execution_count="17"}
``` python
# Year VS all_motor_vehicles

plt.figure(figsize=(15,8))
plt.title('Year VS all_motor_vehicles')
sns.pointplot(x='Year', y='all_motor_vehicles', data=df)
plt.show()
```

::: {.output .display_data}
![](vertopal_febcfee06f0b43d2856ffcbef3b73c40/983a5724b85203f87aca0df5ddc6d0e8599e2f10.png)
:::
:::

::: {.cell .code execution_count="18"}
``` python
# # Source 
# # ONS MYE ,GLA Population Projections

# plt.figure(figsize=(8,5))
# sns.lineplot(x='Year', y='all_motor_vehicles', data=df, hue='Source')
# plt.xticks(list(range(50)))
# plt.show()


plt.style.use("fivethirtyeight")
sns.set_style({'font.sans-serif': ['simhei', 'Arial']})
```
:::

::: {.cell .code execution_count="19"}
``` python
# Name plot 

plt.figure(figsize=[9, 7])
plt.title('Name ratio')
df['Name'].value_counts().plot.pie(autopct='%1.1f%%')
plt.show()
```

::: {.output .display_data}
![](vertopal_febcfee06f0b43d2856ffcbef3b73c40/2c5b166cdc39f69b4d675ea07d500fb9126346f1.png)
:::
:::

::: {.cell .code execution_count="20"}
``` python
## all_motor_vehicles plot

plt.figure(figsize=(9, 6)) 
plt.title('Year and all motor vehicles')
sns.lineplot(data = df[:120],x = df['Year'][:120],y = df['all_motor_vehicles'][:120])
plt.show()
```

::: {.output .display_data}
![](vertopal_febcfee06f0b43d2856ffcbef3b73c40/86e7b4f724afbf806f50b442a19f409ecce42be3.png)
:::
:::

::: {.cell .code execution_count="21"}
``` python
# Source ratio plot 

plt.figure(figsize=[9, 7])
plt.title('Source ratio')
df['Source'].value_counts().plot.pie(autopct='%1.1f%%')
plt.show()
```

::: {.output .display_data}
![](vertopal_febcfee06f0b43d2856ffcbef3b73c40/d54a411dda28c4f0cf83b24247b413f520f2c3fc.png)
:::
:::

::: {.cell .code execution_count="22"}
``` python
# Source  VS  all_motor_vehicles plot 

f, ax3 = plt.subplots(figsize=(5, 6.5))
sns.barplot(x='Source', y='all_motor_vehicles', data=df, ax=ax3)
ax3.set_title('Source and all motor vehicles', fontsize=15)
ax3.set_xlabel('Source ')
ax3.set_ylabel('all motor vehicles')
plt.show()
```

::: {.output .display_data}
![](vertopal_febcfee06f0b43d2856ffcbef3b73c40/f219df009f653713c31258919e914487912d20d5.png)
:::
:::

::: {.cell .code execution_count="23"}
``` python
# cars_and_taxis VS  all_motor_vehicles plot 

plt.figure(figsize=(9, 5)) 
plt.title('cars_and_taxis VS all_motor_vehicles')
sns.scatterplot(data = df,x = df['cars_and_taxis'],y = df['all_motor_vehicles'])
plt.show()
```

::: {.output .display_data}
![](vertopal_febcfee06f0b43d2856ffcbef3b73c40/969313c3f6ad5c3fd1a3fc525ad3b7374d89ef3c.png)
:::
:::

::: {.cell .code execution_count="24"}
``` python
# cars_and_taxis 频率直方图

plt.figure(figsize=(9, 6))  
plt.title("cars_and_taxis distribution")
# sns.scatterplot(x='YearBuilt', y='SalePrice', data=train) 
sns.distplot(df.cars_and_taxis,color = 'r')  # 写法二
plt.show()
```

::: {.output .display_data}
![](vertopal_febcfee06f0b43d2856ffcbef3b73c40/d8a46ef2fb638af5a5831d8b2d42dbc084d43392.png)
:::
:::

::: {.cell .code execution_count="25"}
``` python
# cars_and_taxis VS all_motor_vehicles

plt.figure(figsize=(9, 6))
plt.title("cars_and_taxis VS all_motor_vehicles")
sns.regplot(df['cars_and_taxis'],df['all_motor_vehicles'])
plt.show()
```

::: {.output .display_data}
![](vertopal_febcfee06f0b43d2856ffcbef3b73c40/28383caeee6b3494ad6114dcb4717557e1adc7eb.png)
:::
:::

::: {.cell .code execution_count="26"}
``` python
# 绘制相关性矩阵，查看各字段之间的相关系数。数值越大说明越相关.

matrix = df.corr()
cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)
plt.figure(figsize=(12, 7))
sns.heatmap(matrix, center=0, annot=True, fmt='.2f', square=True, cmap=cmap)
plt.show()
```

::: {.output .display_data}
![](vertopal_febcfee06f0b43d2856ffcbef3b73c40/b41ec2d95e2a06663db053e5bb05657f7ec7d6b1.png)
:::
:::

::: {.cell .markdown}
### 3、建立模型 {#3建立模型}
:::

::: {.cell .code execution_count="27"}
``` python
# 标签编码  

LE = LabelEncoder()
df['Name'] = LE.fit_transform(df['Name'])
df['Source'] = LE.fit_transform(df['Source'])
df['Code'] = LE.fit_transform(df['Code'])
df 
```

::: {.output .execute_result execution_count="27"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Code</th>
      <th>Name</th>
      <th>Year</th>
      <th>Source</th>
      <th>Population</th>
      <th>Inland_Area _Hectares</th>
      <th>Total_Area_Hectares</th>
      <th>Population_per_hectare</th>
      <th>Square_Kilometres</th>
      <th>Population_per_square_kilometre</th>
      <th>cars_and_taxis</th>
      <th>Total weekly income (£)</th>
      <th>Net income before housing costs (£)</th>
      <th>Net income after housing costs (£)</th>
      <th>all_motor_vehicles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>6</td>
      <td>1999</td>
      <td>1</td>
      <td>6581</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>22.7</td>
      <td>2.9</td>
      <td>2266.2</td>
      <td>287107346.8</td>
      <td>630</td>
      <td>480</td>
      <td>450</td>
      <td>3.422533e+08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>6</td>
      <td>2000</td>
      <td>1</td>
      <td>7014</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>24.2</td>
      <td>2.9</td>
      <td>2415.3</td>
      <td>287107346.8</td>
      <td>730</td>
      <td>510</td>
      <td>460</td>
      <td>3.422533e+08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>6</td>
      <td>2001</td>
      <td>1</td>
      <td>7359</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>25.3</td>
      <td>2.9</td>
      <td>2534.1</td>
      <td>604140895.9</td>
      <td>690</td>
      <td>500</td>
      <td>470</td>
      <td>7.048122e+08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>6</td>
      <td>2002</td>
      <td>1</td>
      <td>7280</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>25.1</td>
      <td>2.9</td>
      <td>2506.9</td>
      <td>446328948.9</td>
      <td>540</td>
      <td>420</td>
      <td>380</td>
      <td>5.228484e+08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>6</td>
      <td>2003</td>
      <td>1</td>
      <td>7115</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>24.5</td>
      <td>2.9</td>
      <td>2450.1</td>
      <td>668148077.7</td>
      <td>500</td>
      <td>420</td>
      <td>370</td>
      <td>8.000379e+08</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1867</th>
      <td>35</td>
      <td>28</td>
      <td>2046</td>
      <td>0</td>
      <td>6573194</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>52.4</td>
      <td>1254.2</td>
      <td>5240.8</td>
      <td>856437997.9</td>
      <td>630</td>
      <td>470</td>
      <td>410</td>
      <td>1.055627e+09</td>
    </tr>
    <tr>
      <th>1868</th>
      <td>35</td>
      <td>28</td>
      <td>2047</td>
      <td>0</td>
      <td>6598789</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>52.6</td>
      <td>1254.2</td>
      <td>5261.2</td>
      <td>425162959.2</td>
      <td>710</td>
      <td>490</td>
      <td>460</td>
      <td>5.072792e+08</td>
    </tr>
    <tr>
      <th>1869</th>
      <td>35</td>
      <td>28</td>
      <td>2048</td>
      <td>0</td>
      <td>6622921</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>52.8</td>
      <td>1254.2</td>
      <td>5280.4</td>
      <td>356124062.4</td>
      <td>540</td>
      <td>400</td>
      <td>360</td>
      <td>4.169537e+08</td>
    </tr>
    <tr>
      <th>1870</th>
      <td>35</td>
      <td>28</td>
      <td>2049</td>
      <td>0</td>
      <td>6647527</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>53.0</td>
      <td>1254.2</td>
      <td>5300.1</td>
      <td>784174777.9</td>
      <td>450</td>
      <td>390</td>
      <td>320</td>
      <td>1.001045e+09</td>
    </tr>
    <tr>
      <th>1871</th>
      <td>35</td>
      <td>28</td>
      <td>2050</td>
      <td>0</td>
      <td>6671295</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>53.2</td>
      <td>1254.2</td>
      <td>5319.0</td>
      <td>659300533.2</td>
      <td>490</td>
      <td>360</td>
      <td>310</td>
      <td>8.107151e+08</td>
    </tr>
  </tbody>
</table>
<p>1872 rows × 15 columns</p>
</div>
```
:::
:::

::: {.cell .code execution_count="28"}
``` python
df.columns
```

::: {.output .execute_result execution_count="28"}
    Index(['Code', 'Name', 'Year', 'Source', 'Population', 'Inland_Area _Hectares',
           'Total_Area_Hectares', 'Population_per_hectare', 'Square_Kilometres',
           'Population_per_square_kilometre', 'cars_and_taxis',
           'Total weekly income (£)', 'Net income before housing costs (£)',
           'Net income after housing costs (£)', 'all_motor_vehicles'],
          dtype='object')
:::
:::

::: {.cell .code execution_count="29"}
``` python
# log变换 已做好
def logs(all,colunms):
    for col in colunms:
        all[col] = np.log(all[col]+1)
    return all


# 平方变换 已做好
def squ(res,ls):
    m = res.shape[1]
    for i in ls:
        res = res.assign(newcol = pd.Series((res[i]*res[i]).values))
        res.columns.values[m] = i + '_squ'
        m+=1 
    return res 

log_features = ['Inland_Area _Hectares']

squ_features = ['Population_per_hectare']

df = squ(df,squ_features)
df = logs(df,log_features)
df
```

::: {.output .execute_result execution_count="29"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Code</th>
      <th>Name</th>
      <th>Year</th>
      <th>Source</th>
      <th>Population</th>
      <th>Inland_Area _Hectares</th>
      <th>Total_Area_Hectares</th>
      <th>Population_per_hectare</th>
      <th>Square_Kilometres</th>
      <th>Population_per_square_kilometre</th>
      <th>cars_and_taxis</th>
      <th>Total weekly income (£)</th>
      <th>Net income before housing costs (£)</th>
      <th>Net income after housing costs (£)</th>
      <th>all_motor_vehicles</th>
      <th>Population_per_hectare_squ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>6</td>
      <td>1999</td>
      <td>1</td>
      <td>6581</td>
      <td>5.674697</td>
      <td>314.9</td>
      <td>22.7</td>
      <td>2.9</td>
      <td>2266.2</td>
      <td>287107346.8</td>
      <td>630</td>
      <td>480</td>
      <td>450</td>
      <td>3.422533e+08</td>
      <td>515.29</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>6</td>
      <td>2000</td>
      <td>1</td>
      <td>7014</td>
      <td>5.674697</td>
      <td>314.9</td>
      <td>24.2</td>
      <td>2.9</td>
      <td>2415.3</td>
      <td>287107346.8</td>
      <td>730</td>
      <td>510</td>
      <td>460</td>
      <td>3.422533e+08</td>
      <td>585.64</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>6</td>
      <td>2001</td>
      <td>1</td>
      <td>7359</td>
      <td>5.674697</td>
      <td>314.9</td>
      <td>25.3</td>
      <td>2.9</td>
      <td>2534.1</td>
      <td>604140895.9</td>
      <td>690</td>
      <td>500</td>
      <td>470</td>
      <td>7.048122e+08</td>
      <td>640.09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>6</td>
      <td>2002</td>
      <td>1</td>
      <td>7280</td>
      <td>5.674697</td>
      <td>314.9</td>
      <td>25.1</td>
      <td>2.9</td>
      <td>2506.9</td>
      <td>446328948.9</td>
      <td>540</td>
      <td>420</td>
      <td>380</td>
      <td>5.228484e+08</td>
      <td>630.01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>6</td>
      <td>2003</td>
      <td>1</td>
      <td>7115</td>
      <td>5.674697</td>
      <td>314.9</td>
      <td>24.5</td>
      <td>2.9</td>
      <td>2450.1</td>
      <td>668148077.7</td>
      <td>500</td>
      <td>420</td>
      <td>370</td>
      <td>8.000379e+08</td>
      <td>600.25</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1867</th>
      <td>35</td>
      <td>28</td>
      <td>2046</td>
      <td>0</td>
      <td>6573194</td>
      <td>11.739460</td>
      <td>126675.6</td>
      <td>52.4</td>
      <td>1254.2</td>
      <td>5240.8</td>
      <td>856437997.9</td>
      <td>630</td>
      <td>470</td>
      <td>410</td>
      <td>1.055627e+09</td>
      <td>2745.76</td>
    </tr>
    <tr>
      <th>1868</th>
      <td>35</td>
      <td>28</td>
      <td>2047</td>
      <td>0</td>
      <td>6598789</td>
      <td>11.739460</td>
      <td>126675.6</td>
      <td>52.6</td>
      <td>1254.2</td>
      <td>5261.2</td>
      <td>425162959.2</td>
      <td>710</td>
      <td>490</td>
      <td>460</td>
      <td>5.072792e+08</td>
      <td>2766.76</td>
    </tr>
    <tr>
      <th>1869</th>
      <td>35</td>
      <td>28</td>
      <td>2048</td>
      <td>0</td>
      <td>6622921</td>
      <td>11.739460</td>
      <td>126675.6</td>
      <td>52.8</td>
      <td>1254.2</td>
      <td>5280.4</td>
      <td>356124062.4</td>
      <td>540</td>
      <td>400</td>
      <td>360</td>
      <td>4.169537e+08</td>
      <td>2787.84</td>
    </tr>
    <tr>
      <th>1870</th>
      <td>35</td>
      <td>28</td>
      <td>2049</td>
      <td>0</td>
      <td>6647527</td>
      <td>11.739460</td>
      <td>126675.6</td>
      <td>53.0</td>
      <td>1254.2</td>
      <td>5300.1</td>
      <td>784174777.9</td>
      <td>450</td>
      <td>390</td>
      <td>320</td>
      <td>1.001045e+09</td>
      <td>2809.00</td>
    </tr>
    <tr>
      <th>1871</th>
      <td>35</td>
      <td>28</td>
      <td>2050</td>
      <td>0</td>
      <td>6671295</td>
      <td>11.739460</td>
      <td>126675.6</td>
      <td>53.2</td>
      <td>1254.2</td>
      <td>5319.0</td>
      <td>659300533.2</td>
      <td>490</td>
      <td>360</td>
      <td>310</td>
      <td>8.107151e+08</td>
      <td>2830.24</td>
    </tr>
  </tbody>
</table>
<p>1872 rows × 16 columns</p>
</div>
```
:::
:::

::: {.cell .code execution_count="30"}
``` python
# from sklearn.model_selection import KFold

# folds = KFold(n_splits=5,shuffle=True,random_state=2022)
# for col in df.columns:
#     colname = col+'_kfold'
#     for fold_,(trn_idx,val_idx) in enumerate(folds.split(df,df)):
#         tmp = df.iloc[trn_idx]
#         order_label = tmp.groupby([col])['all_motor_vehicles'].mean()
#         df[colname] = df[col].map(order_label)
# #     order_label = df.groupby([col])['all_motor_vehicles'].mean()
# #     test[colname] = test[col].map(order_label)
# df 
```
:::

::: {.cell .code execution_count="31"}
``` python
# 数据归一化 

features = [fea for fea in df.columns if fea not in ['all_motor_vehicles']]
scaler = MinMaxScaler(feature_range=(0, 1))

for f in features:
    df[f] = scaler.fit_transform(df[f].values.reshape(-1,1))
df 
```

::: {.output .execute_result execution_count="31"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Code</th>
      <th>Name</th>
      <th>Year</th>
      <th>Source</th>
      <th>Population</th>
      <th>Inland_Area _Hectares</th>
      <th>Total_Area_Hectares</th>
      <th>Population_per_hectare</th>
      <th>Square_Kilometres</th>
      <th>Population_per_square_kilometre</th>
      <th>cars_and_taxis</th>
      <th>Total weekly income (£)</th>
      <th>Net income before housing costs (£)</th>
      <th>Net income after housing costs (£)</th>
      <th>all_motor_vehicles</th>
      <th>Population_per_hectare_squ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.016037</td>
      <td>0.000000</td>
      <td>0.015623</td>
      <td>0.038759</td>
      <td>0.323529</td>
      <td>0.446809</td>
      <td>0.479167</td>
      <td>3.422533e+08</td>
      <td>0.002918</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.166667</td>
      <td>0.019608</td>
      <td>1.0</td>
      <td>0.000039</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.023797</td>
      <td>0.000000</td>
      <td>0.023337</td>
      <td>0.038759</td>
      <td>0.421569</td>
      <td>0.510638</td>
      <td>0.500000</td>
      <td>3.422533e+08</td>
      <td>0.004483</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.166667</td>
      <td>0.039216</td>
      <td>1.0</td>
      <td>0.000069</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.029488</td>
      <td>0.000000</td>
      <td>0.029483</td>
      <td>0.081696</td>
      <td>0.382353</td>
      <td>0.489362</td>
      <td>0.520833</td>
      <td>7.048122e+08</td>
      <td>0.005695</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.166667</td>
      <td>0.058824</td>
      <td>1.0</td>
      <td>0.000062</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.028453</td>
      <td>0.000000</td>
      <td>0.028075</td>
      <td>0.060323</td>
      <td>0.235294</td>
      <td>0.319149</td>
      <td>0.333333</td>
      <td>5.228484e+08</td>
      <td>0.005470</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.166667</td>
      <td>0.078431</td>
      <td>1.0</td>
      <td>0.000048</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025349</td>
      <td>0.000000</td>
      <td>0.025137</td>
      <td>0.090365</td>
      <td>0.196078</td>
      <td>0.319149</td>
      <td>0.312500</td>
      <td>8.000379e+08</td>
      <td>0.004808</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1867</th>
      <td>1.0</td>
      <td>0.777778</td>
      <td>0.921569</td>
      <td>0.0</td>
      <td>0.586218</td>
      <td>0.964087</td>
      <td>0.793944</td>
      <td>0.169684</td>
      <td>0.797413</td>
      <td>0.169508</td>
      <td>0.115866</td>
      <td>0.323529</td>
      <td>0.425532</td>
      <td>0.395833</td>
      <td>1.055627e+09</td>
      <td>0.052547</td>
    </tr>
    <tr>
      <th>1868</th>
      <td>1.0</td>
      <td>0.777778</td>
      <td>0.941176</td>
      <td>0.0</td>
      <td>0.588502</td>
      <td>0.964087</td>
      <td>0.793944</td>
      <td>0.170719</td>
      <td>0.797413</td>
      <td>0.170563</td>
      <td>0.057457</td>
      <td>0.401961</td>
      <td>0.468085</td>
      <td>0.500000</td>
      <td>5.072792e+08</td>
      <td>0.053015</td>
    </tr>
    <tr>
      <th>1869</th>
      <td>1.0</td>
      <td>0.777778</td>
      <td>0.960784</td>
      <td>0.0</td>
      <td>0.590657</td>
      <td>0.964087</td>
      <td>0.793944</td>
      <td>0.171754</td>
      <td>0.797413</td>
      <td>0.171556</td>
      <td>0.048107</td>
      <td>0.235294</td>
      <td>0.276596</td>
      <td>0.291667</td>
      <td>4.169537e+08</td>
      <td>0.053484</td>
    </tr>
    <tr>
      <th>1870</th>
      <td>1.0</td>
      <td>0.777778</td>
      <td>0.980392</td>
      <td>0.0</td>
      <td>0.592853</td>
      <td>0.964087</td>
      <td>0.793944</td>
      <td>0.172788</td>
      <td>0.797413</td>
      <td>0.172575</td>
      <td>0.106079</td>
      <td>0.147059</td>
      <td>0.255319</td>
      <td>0.208333</td>
      <td>1.001045e+09</td>
      <td>0.053955</td>
    </tr>
    <tr>
      <th>1871</th>
      <td>1.0</td>
      <td>0.777778</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.594975</td>
      <td>0.964087</td>
      <td>0.793944</td>
      <td>0.173823</td>
      <td>0.797413</td>
      <td>0.173553</td>
      <td>0.089167</td>
      <td>0.186275</td>
      <td>0.191489</td>
      <td>0.187500</td>
      <td>8.107151e+08</td>
      <td>0.054427</td>
    </tr>
  </tbody>
</table>
<p>1872 rows × 16 columns</p>
</div>
```
:::
:::

::: {.cell .code execution_count="32"}
``` python
df = df.fillna(df.mean())
df 
```

::: {.output .execute_result execution_count="32"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Code</th>
      <th>Name</th>
      <th>Year</th>
      <th>Source</th>
      <th>Population</th>
      <th>Inland_Area _Hectares</th>
      <th>Total_Area_Hectares</th>
      <th>Population_per_hectare</th>
      <th>Square_Kilometres</th>
      <th>Population_per_square_kilometre</th>
      <th>cars_and_taxis</th>
      <th>Total weekly income (£)</th>
      <th>Net income before housing costs (£)</th>
      <th>Net income after housing costs (£)</th>
      <th>all_motor_vehicles</th>
      <th>Population_per_hectare_squ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.016037</td>
      <td>0.000000</td>
      <td>0.015623</td>
      <td>0.038759</td>
      <td>0.323529</td>
      <td>0.446809</td>
      <td>0.479167</td>
      <td>3.422533e+08</td>
      <td>0.002918</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.166667</td>
      <td>0.019608</td>
      <td>1.0</td>
      <td>0.000039</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.023797</td>
      <td>0.000000</td>
      <td>0.023337</td>
      <td>0.038759</td>
      <td>0.421569</td>
      <td>0.510638</td>
      <td>0.500000</td>
      <td>3.422533e+08</td>
      <td>0.004483</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.166667</td>
      <td>0.039216</td>
      <td>1.0</td>
      <td>0.000069</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.029488</td>
      <td>0.000000</td>
      <td>0.029483</td>
      <td>0.081696</td>
      <td>0.382353</td>
      <td>0.489362</td>
      <td>0.520833</td>
      <td>7.048122e+08</td>
      <td>0.005695</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.166667</td>
      <td>0.058824</td>
      <td>1.0</td>
      <td>0.000062</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.028453</td>
      <td>0.000000</td>
      <td>0.028075</td>
      <td>0.060323</td>
      <td>0.235294</td>
      <td>0.319149</td>
      <td>0.333333</td>
      <td>5.228484e+08</td>
      <td>0.005470</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.166667</td>
      <td>0.078431</td>
      <td>1.0</td>
      <td>0.000048</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025349</td>
      <td>0.000000</td>
      <td>0.025137</td>
      <td>0.090365</td>
      <td>0.196078</td>
      <td>0.319149</td>
      <td>0.312500</td>
      <td>8.000379e+08</td>
      <td>0.004808</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1867</th>
      <td>1.0</td>
      <td>0.777778</td>
      <td>0.921569</td>
      <td>0.0</td>
      <td>0.586218</td>
      <td>0.964087</td>
      <td>0.793944</td>
      <td>0.169684</td>
      <td>0.797413</td>
      <td>0.169508</td>
      <td>0.115866</td>
      <td>0.323529</td>
      <td>0.425532</td>
      <td>0.395833</td>
      <td>1.055627e+09</td>
      <td>0.052547</td>
    </tr>
    <tr>
      <th>1868</th>
      <td>1.0</td>
      <td>0.777778</td>
      <td>0.941176</td>
      <td>0.0</td>
      <td>0.588502</td>
      <td>0.964087</td>
      <td>0.793944</td>
      <td>0.170719</td>
      <td>0.797413</td>
      <td>0.170563</td>
      <td>0.057457</td>
      <td>0.401961</td>
      <td>0.468085</td>
      <td>0.500000</td>
      <td>5.072792e+08</td>
      <td>0.053015</td>
    </tr>
    <tr>
      <th>1869</th>
      <td>1.0</td>
      <td>0.777778</td>
      <td>0.960784</td>
      <td>0.0</td>
      <td>0.590657</td>
      <td>0.964087</td>
      <td>0.793944</td>
      <td>0.171754</td>
      <td>0.797413</td>
      <td>0.171556</td>
      <td>0.048107</td>
      <td>0.235294</td>
      <td>0.276596</td>
      <td>0.291667</td>
      <td>4.169537e+08</td>
      <td>0.053484</td>
    </tr>
    <tr>
      <th>1870</th>
      <td>1.0</td>
      <td>0.777778</td>
      <td>0.980392</td>
      <td>0.0</td>
      <td>0.592853</td>
      <td>0.964087</td>
      <td>0.793944</td>
      <td>0.172788</td>
      <td>0.797413</td>
      <td>0.172575</td>
      <td>0.106079</td>
      <td>0.147059</td>
      <td>0.255319</td>
      <td>0.208333</td>
      <td>1.001045e+09</td>
      <td>0.053955</td>
    </tr>
    <tr>
      <th>1871</th>
      <td>1.0</td>
      <td>0.777778</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.594975</td>
      <td>0.964087</td>
      <td>0.793944</td>
      <td>0.173823</td>
      <td>0.797413</td>
      <td>0.173553</td>
      <td>0.089167</td>
      <td>0.186275</td>
      <td>0.191489</td>
      <td>0.187500</td>
      <td>8.107151e+08</td>
      <td>0.054427</td>
    </tr>
  </tbody>
</table>
<p>1872 rows × 16 columns</p>
</div>
```
:::
:::

::: {.cell .code execution_count="33"}
``` python
# 划分数据集 

data = df.drop(columns=[
                        'all_motor_vehicles'
                       ])

label = df['all_motor_vehicles']

test_size = 0.2
random_state = 2022 
x_trian,x_test,y_train,y_test = train_test_split(data,
                                                 label,
                                                 test_size = test_size,
                                                 shuffle = False,
                                                 random_state = random_state
                                                )
```
:::

::: {.cell .code execution_count="34"}
``` python
# 线性回归

from sklearn.linear_model import LinearRegression 
LR = LinearRegression(fit_intercept=True,
                      normalize=False,
                      copy_X=True,
                      n_jobs=-2,
                     )

LR = LR.fit(x_trian,y_train)
```
:::

::: {.cell .code execution_count="35"}
``` python
# 测试模型 

pred = LR.predict(x_test)
r2 = r2_score(pred,y_test)
print('LinearRegression r2:',round(r2,5))
```

::: {.output .stream .stdout}
    LinearRegression r2: 0.99362
:::
:::

::: {.cell .code execution_count="36"}
``` python
# 结果对比  

x_test['predict'] = pred 
x_test['True'] = y_test

plt.figure(figsize=(15,8)) 
plt.title('LinearRegression ')
sns.lineplot(data = x_test,x = range(375),y = x_test['predict'],label = 'predict')
sns.lineplot(data = x_test,x = range(375),y = x_test['True'],label = 'True')
plt.legend(loc='best')
plt.show()
```

::: {.output .display_data}
![](vertopal_febcfee06f0b43d2856ffcbef3b73c40/f2af43eeff542ef1644933d023d0f63100b892f6.png)
:::
:::

::: {.cell .code execution_count="37" scrolled="true"}
``` python
df_rf
```

::: {.output .execute_result execution_count="37"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Code</th>
      <th>Name</th>
      <th>Year</th>
      <th>Source</th>
      <th>Population</th>
      <th>Inland_Area _Hectares</th>
      <th>Total_Area_Hectares</th>
      <th>Population_per_hectare</th>
      <th>Square_Kilometres</th>
      <th>Population_per_square_kilometre</th>
      <th>cars_and_taxis</th>
      <th>Total weekly income (£)</th>
      <th>Net income before housing costs (£)</th>
      <th>Net income after housing costs (£)</th>
      <th>all_motor_vehicles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E09000001</td>
      <td>City of London</td>
      <td>1999</td>
      <td>ONS MYE</td>
      <td>6581</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>22.7</td>
      <td>2.9</td>
      <td>2266.2</td>
      <td>287107346.8</td>
      <td>630</td>
      <td>480</td>
      <td>450</td>
      <td>3.422533e+08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E09000001</td>
      <td>City of London</td>
      <td>2000</td>
      <td>ONS MYE</td>
      <td>7014</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>24.2</td>
      <td>2.9</td>
      <td>2415.3</td>
      <td>287107346.8</td>
      <td>730</td>
      <td>510</td>
      <td>460</td>
      <td>3.422533e+08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E09000001</td>
      <td>City of London</td>
      <td>2001</td>
      <td>ONS MYE</td>
      <td>7359</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>25.3</td>
      <td>2.9</td>
      <td>2534.1</td>
      <td>604140895.9</td>
      <td>690</td>
      <td>500</td>
      <td>470</td>
      <td>7.048122e+08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>E09000001</td>
      <td>City of London</td>
      <td>2002</td>
      <td>ONS MYE</td>
      <td>7280</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>25.1</td>
      <td>2.9</td>
      <td>2506.9</td>
      <td>446328948.9</td>
      <td>540</td>
      <td>420</td>
      <td>380</td>
      <td>5.228484e+08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E09000001</td>
      <td>City of London</td>
      <td>2003</td>
      <td>ONS MYE</td>
      <td>7115</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>24.5</td>
      <td>2.9</td>
      <td>2450.1</td>
      <td>668148077.7</td>
      <td>500</td>
      <td>420</td>
      <td>370</td>
      <td>8.000379e+08</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1867</th>
      <td>E13000002</td>
      <td>Outer London</td>
      <td>2046</td>
      <td>GLA Population Projections</td>
      <td>6573194</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>52.4</td>
      <td>1254.2</td>
      <td>5240.8</td>
      <td>856437997.9</td>
      <td>630</td>
      <td>470</td>
      <td>410</td>
      <td>1.055627e+09</td>
    </tr>
    <tr>
      <th>1868</th>
      <td>E13000002</td>
      <td>Outer London</td>
      <td>2047</td>
      <td>GLA Population Projections</td>
      <td>6598789</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>52.6</td>
      <td>1254.2</td>
      <td>5261.2</td>
      <td>425162959.2</td>
      <td>710</td>
      <td>490</td>
      <td>460</td>
      <td>5.072792e+08</td>
    </tr>
    <tr>
      <th>1869</th>
      <td>E13000002</td>
      <td>Outer London</td>
      <td>2048</td>
      <td>GLA Population Projections</td>
      <td>6622921</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>52.8</td>
      <td>1254.2</td>
      <td>5280.4</td>
      <td>356124062.4</td>
      <td>540</td>
      <td>400</td>
      <td>360</td>
      <td>4.169537e+08</td>
    </tr>
    <tr>
      <th>1870</th>
      <td>E13000002</td>
      <td>Outer London</td>
      <td>2049</td>
      <td>GLA Population Projections</td>
      <td>6647527</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>53.0</td>
      <td>1254.2</td>
      <td>5300.1</td>
      <td>784174777.9</td>
      <td>450</td>
      <td>390</td>
      <td>320</td>
      <td>1.001045e+09</td>
    </tr>
    <tr>
      <th>1871</th>
      <td>E13000002</td>
      <td>Outer London</td>
      <td>2050</td>
      <td>GLA Population Projections</td>
      <td>6671295</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>53.2</td>
      <td>1254.2</td>
      <td>5319.0</td>
      <td>659300533.2</td>
      <td>490</td>
      <td>360</td>
      <td>310</td>
      <td>8.107151e+08</td>
    </tr>
  </tbody>
</table>
<p>1872 rows × 15 columns</p>
</div>
```
:::
:::

::: {.cell .code execution_count="38" scrolled="true"}
``` python
# 标签编码  

LE = LabelEncoder()
df_rf['Name'] = LE.fit_transform(df_rf['Name'])
df_rf['Source'] = LE.fit_transform(df_rf['Source'])
df_rf['Code'] = LE.fit_transform(df_rf['Code'])
df_rf 
```

::: {.output .execute_result execution_count="38"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Code</th>
      <th>Name</th>
      <th>Year</th>
      <th>Source</th>
      <th>Population</th>
      <th>Inland_Area _Hectares</th>
      <th>Total_Area_Hectares</th>
      <th>Population_per_hectare</th>
      <th>Square_Kilometres</th>
      <th>Population_per_square_kilometre</th>
      <th>cars_and_taxis</th>
      <th>Total weekly income (£)</th>
      <th>Net income before housing costs (£)</th>
      <th>Net income after housing costs (£)</th>
      <th>all_motor_vehicles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>6</td>
      <td>1999</td>
      <td>1</td>
      <td>6581</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>22.7</td>
      <td>2.9</td>
      <td>2266.2</td>
      <td>287107346.8</td>
      <td>630</td>
      <td>480</td>
      <td>450</td>
      <td>3.422533e+08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>6</td>
      <td>2000</td>
      <td>1</td>
      <td>7014</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>24.2</td>
      <td>2.9</td>
      <td>2415.3</td>
      <td>287107346.8</td>
      <td>730</td>
      <td>510</td>
      <td>460</td>
      <td>3.422533e+08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>6</td>
      <td>2001</td>
      <td>1</td>
      <td>7359</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>25.3</td>
      <td>2.9</td>
      <td>2534.1</td>
      <td>604140895.9</td>
      <td>690</td>
      <td>500</td>
      <td>470</td>
      <td>7.048122e+08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>6</td>
      <td>2002</td>
      <td>1</td>
      <td>7280</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>25.1</td>
      <td>2.9</td>
      <td>2506.9</td>
      <td>446328948.9</td>
      <td>540</td>
      <td>420</td>
      <td>380</td>
      <td>5.228484e+08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>6</td>
      <td>2003</td>
      <td>1</td>
      <td>7115</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>24.5</td>
      <td>2.9</td>
      <td>2450.1</td>
      <td>668148077.7</td>
      <td>500</td>
      <td>420</td>
      <td>370</td>
      <td>8.000379e+08</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1867</th>
      <td>35</td>
      <td>28</td>
      <td>2046</td>
      <td>0</td>
      <td>6573194</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>52.4</td>
      <td>1254.2</td>
      <td>5240.8</td>
      <td>856437997.9</td>
      <td>630</td>
      <td>470</td>
      <td>410</td>
      <td>1.055627e+09</td>
    </tr>
    <tr>
      <th>1868</th>
      <td>35</td>
      <td>28</td>
      <td>2047</td>
      <td>0</td>
      <td>6598789</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>52.6</td>
      <td>1254.2</td>
      <td>5261.2</td>
      <td>425162959.2</td>
      <td>710</td>
      <td>490</td>
      <td>460</td>
      <td>5.072792e+08</td>
    </tr>
    <tr>
      <th>1869</th>
      <td>35</td>
      <td>28</td>
      <td>2048</td>
      <td>0</td>
      <td>6622921</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>52.8</td>
      <td>1254.2</td>
      <td>5280.4</td>
      <td>356124062.4</td>
      <td>540</td>
      <td>400</td>
      <td>360</td>
      <td>4.169537e+08</td>
    </tr>
    <tr>
      <th>1870</th>
      <td>35</td>
      <td>28</td>
      <td>2049</td>
      <td>0</td>
      <td>6647527</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>53.0</td>
      <td>1254.2</td>
      <td>5300.1</td>
      <td>784174777.9</td>
      <td>450</td>
      <td>390</td>
      <td>320</td>
      <td>1.001045e+09</td>
    </tr>
    <tr>
      <th>1871</th>
      <td>35</td>
      <td>28</td>
      <td>2050</td>
      <td>0</td>
      <td>6671295</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>53.2</td>
      <td>1254.2</td>
      <td>5319.0</td>
      <td>659300533.2</td>
      <td>490</td>
      <td>360</td>
      <td>310</td>
      <td>8.107151e+08</td>
    </tr>
  </tbody>
</table>
<p>1872 rows × 15 columns</p>
</div>
```
:::
:::

::: {.cell .code execution_count="39"}
``` python
# log变换 已做好
def logs(all,colunms):
    for col in colunms:
        all[col] = np.log(all[col]+1)
    return all


# 平方变换 已做好
def squ(res,ls):
    m = res.shape[1]
    for i in ls:
        res = res.assign(newcol = pd.Series((res[i]*res[i]).values))
        res.columns.values[m] = i + '_squ'
        m+=1 
    return res 

log_features = ['Inland_Area _Hectares']

squ_features = ['Population_per_hectare',]

df_rf = squ(df_rf,squ_features)
df_rf = logs(df_rf,log_features)
df_rf
```

::: {.output .execute_result execution_count="39"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Code</th>
      <th>Name</th>
      <th>Year</th>
      <th>Source</th>
      <th>Population</th>
      <th>Inland_Area _Hectares</th>
      <th>Total_Area_Hectares</th>
      <th>Population_per_hectare</th>
      <th>Square_Kilometres</th>
      <th>Population_per_square_kilometre</th>
      <th>cars_and_taxis</th>
      <th>Total weekly income (£)</th>
      <th>Net income before housing costs (£)</th>
      <th>Net income after housing costs (£)</th>
      <th>all_motor_vehicles</th>
      <th>Population_per_hectare_squ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>6</td>
      <td>1999</td>
      <td>1</td>
      <td>6581</td>
      <td>5.674697</td>
      <td>314.9</td>
      <td>22.7</td>
      <td>2.9</td>
      <td>2266.2</td>
      <td>287107346.8</td>
      <td>630</td>
      <td>480</td>
      <td>450</td>
      <td>3.422533e+08</td>
      <td>515.29</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>6</td>
      <td>2000</td>
      <td>1</td>
      <td>7014</td>
      <td>5.674697</td>
      <td>314.9</td>
      <td>24.2</td>
      <td>2.9</td>
      <td>2415.3</td>
      <td>287107346.8</td>
      <td>730</td>
      <td>510</td>
      <td>460</td>
      <td>3.422533e+08</td>
      <td>585.64</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>6</td>
      <td>2001</td>
      <td>1</td>
      <td>7359</td>
      <td>5.674697</td>
      <td>314.9</td>
      <td>25.3</td>
      <td>2.9</td>
      <td>2534.1</td>
      <td>604140895.9</td>
      <td>690</td>
      <td>500</td>
      <td>470</td>
      <td>7.048122e+08</td>
      <td>640.09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>6</td>
      <td>2002</td>
      <td>1</td>
      <td>7280</td>
      <td>5.674697</td>
      <td>314.9</td>
      <td>25.1</td>
      <td>2.9</td>
      <td>2506.9</td>
      <td>446328948.9</td>
      <td>540</td>
      <td>420</td>
      <td>380</td>
      <td>5.228484e+08</td>
      <td>630.01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>6</td>
      <td>2003</td>
      <td>1</td>
      <td>7115</td>
      <td>5.674697</td>
      <td>314.9</td>
      <td>24.5</td>
      <td>2.9</td>
      <td>2450.1</td>
      <td>668148077.7</td>
      <td>500</td>
      <td>420</td>
      <td>370</td>
      <td>8.000379e+08</td>
      <td>600.25</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1867</th>
      <td>35</td>
      <td>28</td>
      <td>2046</td>
      <td>0</td>
      <td>6573194</td>
      <td>11.739460</td>
      <td>126675.6</td>
      <td>52.4</td>
      <td>1254.2</td>
      <td>5240.8</td>
      <td>856437997.9</td>
      <td>630</td>
      <td>470</td>
      <td>410</td>
      <td>1.055627e+09</td>
      <td>2745.76</td>
    </tr>
    <tr>
      <th>1868</th>
      <td>35</td>
      <td>28</td>
      <td>2047</td>
      <td>0</td>
      <td>6598789</td>
      <td>11.739460</td>
      <td>126675.6</td>
      <td>52.6</td>
      <td>1254.2</td>
      <td>5261.2</td>
      <td>425162959.2</td>
      <td>710</td>
      <td>490</td>
      <td>460</td>
      <td>5.072792e+08</td>
      <td>2766.76</td>
    </tr>
    <tr>
      <th>1869</th>
      <td>35</td>
      <td>28</td>
      <td>2048</td>
      <td>0</td>
      <td>6622921</td>
      <td>11.739460</td>
      <td>126675.6</td>
      <td>52.8</td>
      <td>1254.2</td>
      <td>5280.4</td>
      <td>356124062.4</td>
      <td>540</td>
      <td>400</td>
      <td>360</td>
      <td>4.169537e+08</td>
      <td>2787.84</td>
    </tr>
    <tr>
      <th>1870</th>
      <td>35</td>
      <td>28</td>
      <td>2049</td>
      <td>0</td>
      <td>6647527</td>
      <td>11.739460</td>
      <td>126675.6</td>
      <td>53.0</td>
      <td>1254.2</td>
      <td>5300.1</td>
      <td>784174777.9</td>
      <td>450</td>
      <td>390</td>
      <td>320</td>
      <td>1.001045e+09</td>
      <td>2809.00</td>
    </tr>
    <tr>
      <th>1871</th>
      <td>35</td>
      <td>28</td>
      <td>2050</td>
      <td>0</td>
      <td>6671295</td>
      <td>11.739460</td>
      <td>126675.6</td>
      <td>53.2</td>
      <td>1254.2</td>
      <td>5319.0</td>
      <td>659300533.2</td>
      <td>490</td>
      <td>360</td>
      <td>310</td>
      <td>8.107151e+08</td>
      <td>2830.24</td>
    </tr>
  </tbody>
</table>
<p>1872 rows × 16 columns</p>
</div>
```
:::
:::

::: {.cell .code execution_count="40"}
``` python
# 划分数据集 

data = df_rf.drop(columns=[
                        'all_motor_vehicles'
                           ])

label = df_rf['all_motor_vehicles']

test_size = 0.2
random_state = 2022 
x_trian,x_test,y_train,y_test = train_test_split(data,
                                                 label,
                                                 test_size = test_size,
                                                 shuffle = False,
                                                 random_state = random_state
                                                )
```
:::

::: {.cell .code execution_count="41"}
``` python
## 随机森林 

randomforest = RandomForestRegressor(n_estimators=500,
                                        max_depth=10,
                                        min_samples_split=2,
                                        min_samples_leaf=1,
                                        min_weight_fraction_leaf=0.0,
                                        max_features='auto',
                                        max_leaf_nodes=None,
                                        min_impurity_decrease=0.0,
                                        min_impurity_split=None,
                                        bootstrap=True,
                                        oob_score=False,
                                        n_jobs=None,
                                        random_state=None,
                                        verbose=0,
                                        warm_start=False,
                                        ccp_alpha=0.0,
                                        max_samples=None,
                                     )

randomforest = randomforest.fit(x_trian,y_train)

```
:::

::: {.cell .code execution_count="42"}
``` python
# 测试模型 

# x_test = x_test.drop(columns=['predict','True'])
pred = randomforest.predict(x_test)
r2 = r2_score(pred,y_test)
print('RandomForest r2:',round(r2,5))
```

::: {.output .stream .stdout}
    RandomForest r2: 0.99806
:::
:::

::: {.cell .code execution_count="43"}
``` python
# 结果对比  

x_test['predict'] = pred 
x_test['True'] = y_test

plt.figure(figsize=(15,8)) 
plt.title('RandomForest ')
sns.lineplot(data = x_test,x = range(375),y = x_test['predict'],label = 'predict')
sns.lineplot(data = x_test,x = range(375),y = x_test['True'],label = 'True')
plt.legend(loc='best')
plt.show()
```

::: {.output .display_data}
![](vertopal_febcfee06f0b43d2856ffcbef3b73c40/f9880c7ff4785b336e799ec6e8b1ea88e220a62a.png)
:::
:::

::: {.cell .code execution_count="44" scrolled="true"}
``` python
df_mlp
```

::: {.output .execute_result execution_count="44"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Code</th>
      <th>Name</th>
      <th>Year</th>
      <th>Source</th>
      <th>Population</th>
      <th>Inland_Area _Hectares</th>
      <th>Total_Area_Hectares</th>
      <th>Population_per_hectare</th>
      <th>Square_Kilometres</th>
      <th>Population_per_square_kilometre</th>
      <th>cars_and_taxis</th>
      <th>Total weekly income (£)</th>
      <th>Net income before housing costs (£)</th>
      <th>Net income after housing costs (£)</th>
      <th>all_motor_vehicles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E09000001</td>
      <td>City of London</td>
      <td>1999</td>
      <td>ONS MYE</td>
      <td>6581</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>22.7</td>
      <td>2.9</td>
      <td>2266.2</td>
      <td>287107346.8</td>
      <td>630</td>
      <td>480</td>
      <td>450</td>
      <td>3.422533e+08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E09000001</td>
      <td>City of London</td>
      <td>2000</td>
      <td>ONS MYE</td>
      <td>7014</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>24.2</td>
      <td>2.9</td>
      <td>2415.3</td>
      <td>287107346.8</td>
      <td>730</td>
      <td>510</td>
      <td>460</td>
      <td>3.422533e+08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E09000001</td>
      <td>City of London</td>
      <td>2001</td>
      <td>ONS MYE</td>
      <td>7359</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>25.3</td>
      <td>2.9</td>
      <td>2534.1</td>
      <td>604140895.9</td>
      <td>690</td>
      <td>500</td>
      <td>470</td>
      <td>7.048122e+08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>E09000001</td>
      <td>City of London</td>
      <td>2002</td>
      <td>ONS MYE</td>
      <td>7280</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>25.1</td>
      <td>2.9</td>
      <td>2506.9</td>
      <td>446328948.9</td>
      <td>540</td>
      <td>420</td>
      <td>380</td>
      <td>5.228484e+08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E09000001</td>
      <td>City of London</td>
      <td>2003</td>
      <td>ONS MYE</td>
      <td>7115</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>24.5</td>
      <td>2.9</td>
      <td>2450.1</td>
      <td>668148077.7</td>
      <td>500</td>
      <td>420</td>
      <td>370</td>
      <td>8.000379e+08</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1867</th>
      <td>E13000002</td>
      <td>Outer London</td>
      <td>2046</td>
      <td>GLA Population Projections</td>
      <td>6573194</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>52.4</td>
      <td>1254.2</td>
      <td>5240.8</td>
      <td>856437997.9</td>
      <td>630</td>
      <td>470</td>
      <td>410</td>
      <td>1.055627e+09</td>
    </tr>
    <tr>
      <th>1868</th>
      <td>E13000002</td>
      <td>Outer London</td>
      <td>2047</td>
      <td>GLA Population Projections</td>
      <td>6598789</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>52.6</td>
      <td>1254.2</td>
      <td>5261.2</td>
      <td>425162959.2</td>
      <td>710</td>
      <td>490</td>
      <td>460</td>
      <td>5.072792e+08</td>
    </tr>
    <tr>
      <th>1869</th>
      <td>E13000002</td>
      <td>Outer London</td>
      <td>2048</td>
      <td>GLA Population Projections</td>
      <td>6622921</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>52.8</td>
      <td>1254.2</td>
      <td>5280.4</td>
      <td>356124062.4</td>
      <td>540</td>
      <td>400</td>
      <td>360</td>
      <td>4.169537e+08</td>
    </tr>
    <tr>
      <th>1870</th>
      <td>E13000002</td>
      <td>Outer London</td>
      <td>2049</td>
      <td>GLA Population Projections</td>
      <td>6647527</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>53.0</td>
      <td>1254.2</td>
      <td>5300.1</td>
      <td>784174777.9</td>
      <td>450</td>
      <td>390</td>
      <td>320</td>
      <td>1.001045e+09</td>
    </tr>
    <tr>
      <th>1871</th>
      <td>E13000002</td>
      <td>Outer London</td>
      <td>2050</td>
      <td>GLA Population Projections</td>
      <td>6671295</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>53.2</td>
      <td>1254.2</td>
      <td>5319.0</td>
      <td>659300533.2</td>
      <td>490</td>
      <td>360</td>
      <td>310</td>
      <td>8.107151e+08</td>
    </tr>
  </tbody>
</table>
<p>1872 rows × 15 columns</p>
</div>
```
:::
:::

::: {.cell .code execution_count="45"}
``` python
# 标签编码  

LE = LabelEncoder()
df_mlp['Name'] = LE.fit_transform(df_mlp['Name'])
df_mlp['Source'] = LE.fit_transform(df_mlp['Source'])
df_mlp['Code'] = LE.fit_transform(df_mlp['Code'])
df_mlp 
```

::: {.output .execute_result execution_count="45"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Code</th>
      <th>Name</th>
      <th>Year</th>
      <th>Source</th>
      <th>Population</th>
      <th>Inland_Area _Hectares</th>
      <th>Total_Area_Hectares</th>
      <th>Population_per_hectare</th>
      <th>Square_Kilometres</th>
      <th>Population_per_square_kilometre</th>
      <th>cars_and_taxis</th>
      <th>Total weekly income (£)</th>
      <th>Net income before housing costs (£)</th>
      <th>Net income after housing costs (£)</th>
      <th>all_motor_vehicles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>6</td>
      <td>1999</td>
      <td>1</td>
      <td>6581</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>22.7</td>
      <td>2.9</td>
      <td>2266.2</td>
      <td>287107346.8</td>
      <td>630</td>
      <td>480</td>
      <td>450</td>
      <td>3.422533e+08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>6</td>
      <td>2000</td>
      <td>1</td>
      <td>7014</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>24.2</td>
      <td>2.9</td>
      <td>2415.3</td>
      <td>287107346.8</td>
      <td>730</td>
      <td>510</td>
      <td>460</td>
      <td>3.422533e+08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>6</td>
      <td>2001</td>
      <td>1</td>
      <td>7359</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>25.3</td>
      <td>2.9</td>
      <td>2534.1</td>
      <td>604140895.9</td>
      <td>690</td>
      <td>500</td>
      <td>470</td>
      <td>7.048122e+08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>6</td>
      <td>2002</td>
      <td>1</td>
      <td>7280</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>25.1</td>
      <td>2.9</td>
      <td>2506.9</td>
      <td>446328948.9</td>
      <td>540</td>
      <td>420</td>
      <td>380</td>
      <td>5.228484e+08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>6</td>
      <td>2003</td>
      <td>1</td>
      <td>7115</td>
      <td>290.4</td>
      <td>314.9</td>
      <td>24.5</td>
      <td>2.9</td>
      <td>2450.1</td>
      <td>668148077.7</td>
      <td>500</td>
      <td>420</td>
      <td>370</td>
      <td>8.000379e+08</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1867</th>
      <td>35</td>
      <td>28</td>
      <td>2046</td>
      <td>0</td>
      <td>6573194</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>52.4</td>
      <td>1254.2</td>
      <td>5240.8</td>
      <td>856437997.9</td>
      <td>630</td>
      <td>470</td>
      <td>410</td>
      <td>1.055627e+09</td>
    </tr>
    <tr>
      <th>1868</th>
      <td>35</td>
      <td>28</td>
      <td>2047</td>
      <td>0</td>
      <td>6598789</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>52.6</td>
      <td>1254.2</td>
      <td>5261.2</td>
      <td>425162959.2</td>
      <td>710</td>
      <td>490</td>
      <td>460</td>
      <td>5.072792e+08</td>
    </tr>
    <tr>
      <th>1869</th>
      <td>35</td>
      <td>28</td>
      <td>2048</td>
      <td>0</td>
      <td>6622921</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>52.8</td>
      <td>1254.2</td>
      <td>5280.4</td>
      <td>356124062.4</td>
      <td>540</td>
      <td>400</td>
      <td>360</td>
      <td>4.169537e+08</td>
    </tr>
    <tr>
      <th>1870</th>
      <td>35</td>
      <td>28</td>
      <td>2049</td>
      <td>0</td>
      <td>6647527</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>53.0</td>
      <td>1254.2</td>
      <td>5300.1</td>
      <td>784174777.9</td>
      <td>450</td>
      <td>390</td>
      <td>320</td>
      <td>1.001045e+09</td>
    </tr>
    <tr>
      <th>1871</th>
      <td>35</td>
      <td>28</td>
      <td>2050</td>
      <td>0</td>
      <td>6671295</td>
      <td>125423.6</td>
      <td>126675.6</td>
      <td>53.2</td>
      <td>1254.2</td>
      <td>5319.0</td>
      <td>659300533.2</td>
      <td>490</td>
      <td>360</td>
      <td>310</td>
      <td>8.107151e+08</td>
    </tr>
  </tbody>
</table>
<p>1872 rows × 15 columns</p>
</div>
```
:::
:::

::: {.cell .code execution_count="46" scrolled="true"}
``` python
# log变换 已做好
def logs(all,colunms):
    for col in colunms:
        all[col] = np.log(all[col]+1)
    return all


# 平方变换 已做好
def squ(res,ls):
    m = res.shape[1]
    for i in ls:
        res = res.assign(newcol = pd.Series((res[i]*res[i]).values))
        res.columns.values[m] = i + '_squ'
        m+=1 
    return res 

log_features = ['Inland_Area _Hectares']

squ_features = ['Population_per_hectare',]

df_mlp = squ(df_mlp,squ_features)
df_mlp = logs(df_mlp,log_features)
df_mlp
```

::: {.output .execute_result execution_count="46"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Code</th>
      <th>Name</th>
      <th>Year</th>
      <th>Source</th>
      <th>Population</th>
      <th>Inland_Area _Hectares</th>
      <th>Total_Area_Hectares</th>
      <th>Population_per_hectare</th>
      <th>Square_Kilometres</th>
      <th>Population_per_square_kilometre</th>
      <th>cars_and_taxis</th>
      <th>Total weekly income (£)</th>
      <th>Net income before housing costs (£)</th>
      <th>Net income after housing costs (£)</th>
      <th>all_motor_vehicles</th>
      <th>Population_per_hectare_squ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>6</td>
      <td>1999</td>
      <td>1</td>
      <td>6581</td>
      <td>5.674697</td>
      <td>314.9</td>
      <td>22.7</td>
      <td>2.9</td>
      <td>2266.2</td>
      <td>287107346.8</td>
      <td>630</td>
      <td>480</td>
      <td>450</td>
      <td>3.422533e+08</td>
      <td>515.29</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>6</td>
      <td>2000</td>
      <td>1</td>
      <td>7014</td>
      <td>5.674697</td>
      <td>314.9</td>
      <td>24.2</td>
      <td>2.9</td>
      <td>2415.3</td>
      <td>287107346.8</td>
      <td>730</td>
      <td>510</td>
      <td>460</td>
      <td>3.422533e+08</td>
      <td>585.64</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>6</td>
      <td>2001</td>
      <td>1</td>
      <td>7359</td>
      <td>5.674697</td>
      <td>314.9</td>
      <td>25.3</td>
      <td>2.9</td>
      <td>2534.1</td>
      <td>604140895.9</td>
      <td>690</td>
      <td>500</td>
      <td>470</td>
      <td>7.048122e+08</td>
      <td>640.09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>6</td>
      <td>2002</td>
      <td>1</td>
      <td>7280</td>
      <td>5.674697</td>
      <td>314.9</td>
      <td>25.1</td>
      <td>2.9</td>
      <td>2506.9</td>
      <td>446328948.9</td>
      <td>540</td>
      <td>420</td>
      <td>380</td>
      <td>5.228484e+08</td>
      <td>630.01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>6</td>
      <td>2003</td>
      <td>1</td>
      <td>7115</td>
      <td>5.674697</td>
      <td>314.9</td>
      <td>24.5</td>
      <td>2.9</td>
      <td>2450.1</td>
      <td>668148077.7</td>
      <td>500</td>
      <td>420</td>
      <td>370</td>
      <td>8.000379e+08</td>
      <td>600.25</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1867</th>
      <td>35</td>
      <td>28</td>
      <td>2046</td>
      <td>0</td>
      <td>6573194</td>
      <td>11.739460</td>
      <td>126675.6</td>
      <td>52.4</td>
      <td>1254.2</td>
      <td>5240.8</td>
      <td>856437997.9</td>
      <td>630</td>
      <td>470</td>
      <td>410</td>
      <td>1.055627e+09</td>
      <td>2745.76</td>
    </tr>
    <tr>
      <th>1868</th>
      <td>35</td>
      <td>28</td>
      <td>2047</td>
      <td>0</td>
      <td>6598789</td>
      <td>11.739460</td>
      <td>126675.6</td>
      <td>52.6</td>
      <td>1254.2</td>
      <td>5261.2</td>
      <td>425162959.2</td>
      <td>710</td>
      <td>490</td>
      <td>460</td>
      <td>5.072792e+08</td>
      <td>2766.76</td>
    </tr>
    <tr>
      <th>1869</th>
      <td>35</td>
      <td>28</td>
      <td>2048</td>
      <td>0</td>
      <td>6622921</td>
      <td>11.739460</td>
      <td>126675.6</td>
      <td>52.8</td>
      <td>1254.2</td>
      <td>5280.4</td>
      <td>356124062.4</td>
      <td>540</td>
      <td>400</td>
      <td>360</td>
      <td>4.169537e+08</td>
      <td>2787.84</td>
    </tr>
    <tr>
      <th>1870</th>
      <td>35</td>
      <td>28</td>
      <td>2049</td>
      <td>0</td>
      <td>6647527</td>
      <td>11.739460</td>
      <td>126675.6</td>
      <td>53.0</td>
      <td>1254.2</td>
      <td>5300.1</td>
      <td>784174777.9</td>
      <td>450</td>
      <td>390</td>
      <td>320</td>
      <td>1.001045e+09</td>
      <td>2809.00</td>
    </tr>
    <tr>
      <th>1871</th>
      <td>35</td>
      <td>28</td>
      <td>2050</td>
      <td>0</td>
      <td>6671295</td>
      <td>11.739460</td>
      <td>126675.6</td>
      <td>53.2</td>
      <td>1254.2</td>
      <td>5319.0</td>
      <td>659300533.2</td>
      <td>490</td>
      <td>360</td>
      <td>310</td>
      <td>8.107151e+08</td>
      <td>2830.24</td>
    </tr>
  </tbody>
</table>
<p>1872 rows × 16 columns</p>
</div>
```
:::
:::

::: {.cell .code execution_count="47"}
``` python
# 划分数据集 

data = df_mlp.drop(columns=[
                        'all_motor_vehicles'
                           ])

label = df_mlp['all_motor_vehicles']

test_size = 0.2
random_state = 2022 
x_trian,x_test,y_train,y_test = train_test_split(data,
                                                 label,
                                                 test_size = test_size,
                                                 shuffle = False,
                                                 random_state = random_state
                                                )
```
:::

::: {.cell .code execution_count="48"}
``` python
# MLP 

from sklearn.neural_network import MLPRegressor
MLP = MLPRegressor(
                    hidden_layer_sizes=(100,50),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    batch_size='auto',
                    learning_rate='constant',
                    learning_rate_init=0.001,
                    power_t=0.5,
                    max_iter=500,
                    shuffle=True,
                    random_state=None,
                    tol=0.0001,
                    verbose=False,
                    warm_start=False,
                    momentum=0.9,
                    nesterovs_momentum=True,
                    early_stopping=False,
                    validation_fraction=0.1,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-08,
                    n_iter_no_change=10,
                    max_fun=15000,
                    )

    
MLP.fit(x_trian,y_train)
```

::: {.output .execute_result execution_count="48"}
    MLPRegressor(alpha=0.001, hidden_layer_sizes=(100, 50), max_iter=500)
:::
:::

::: {.cell .code execution_count="49"}
``` python
# 测试模型 

# x_test = x_test.drop(columns=['predict','True'])
pred = MLP.predict(x_test)
```
:::

::: {.cell .code execution_count="50"}
``` python
# 计算R2 
r2 = r2_score(pred,y_test)
print('MLP r2:',round(r2,5))
```

::: {.output .stream .stdout}
    MLP r2: 0.99844
:::
:::

::: {.cell .code execution_count="51"}
``` python
# 结果对比  

x_test['predict'] = pred 
x_test['True'] = y_test

plt.figure(figsize=(15,8)) 
plt.title('MLP ')
sns.lineplot(data = x_test,x = range(375),y = x_test['predict'],label = 'predict')
sns.lineplot(data = x_test,x = range(375),y = x_test['True'],label = 'True')
plt.legend(loc='best')
plt.show()
```

::: {.output .display_data}
![](vertopal_febcfee06f0b43d2856ffcbef3b73c40/fbe088ec5dc5e7e4b71abed19c5b705a55a4e0b5.png)
:::
:::

::: {.cell .code}
``` python
```
:::

::: {.cell .code}
``` python
```
:::

::: {.cell .code}
``` python
```
:::

::: {.cell .code}
``` python
```
:::
