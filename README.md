# Mobile price classification with multilayer perceptron

## Authors
 - [Erik Matovič](https://github.com/Matovic)
 
## Usage

[Install tensorflow to enable GPU](https://www.tensorflow.org/install/pip)   

## Solution
### 1. Exploratory Data Analysis

[Dataset](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification?select=train.csv)

Attributes:
 - battery_power - Total energy a battery can store in one time measured in mAh. Range: 501-1998, int64
 - blue - Has bluetooth or not. Range: 0-1, int64
 - clock_speed - speed at which microprocessor executes instructions. Range: 0.5-3, float64
 - dual_sim - Has dual sim support or not. Range: 0-1, int64
 - fc - Front Camera mega pixels. Range: 0-19, int64
 - four_g - Has 4G or not. Range: 0-1, int64
 - int_memory - Internal Memory in Gigabytes. Range: 2-64, int64
 - m_dep - Mobile Depth in cm. Range: 0.1-1, float64
 - mobile_wt - Weight of mobile phone. Range: 80-200, int64
 - n_cores - Number of cores of processor. Range: 1-8, int64
 - pc - Primary Camera mega pixels. Range: 0-20, int64
 - px_height - Pixel Resolution Height. Range: 0-1960, int64
 - px_width - Pixel Resolution Width. Range: 500-1998, int64
 - ram - Random Access Memory in Mega Bytes. Range: 256-3998, int64
 - sc_h - Screen Height of mobile in cm. Range: 5-19, int64
 - sc_w - Screen Width of mobile in cm. Range: 0-18, int64
 - talk_time - longest time that a single battery charge will last. Range: 2-20, int64
 - three_g - Has 3G or not. Range: 0-1, int64
 - touch_screen - Has touch screen or not. Range: 0-1, int64
 - wifi - Has wifi or not. Range: 0-1, int64
 - price_range - This is the target variable with value of 0(low cost), 1(medium cost), 2(high cost) and 3(very high cost). Range: 0-3, int64

### 2. Data Preprocessing

Based on [exploratory data analysis](EDA.ipynb) test set does not have target variable price_range. We split our dataset into train-dev-test. We have train and test sets, but we split test set by half to dev-test sets. We will rougly have train-dev-test 67%-16.5%-16.5%.  


Check if tensorflow detects GPU:
```python3
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))
```


