#!/usr/bin/env python
# coding: utf-8


import os                            
import numpy   as np                 
import pandas  as pd                
from matplotlib import pyplot as plt 
import seaborn as sns                
sns.set()                           
import tensorflow as tf                                                   
from   tensorflow.keras.preprocessing.sequence import TimeseriesGenerator 


train_start_date = "2015-04-28"
train_end_date   = "2017-12-31"
val_start_date   = "2018-01-03"
val_end_date     = "2018-12-31"
test_start_date  = "2019-01-02"
test_end_date    = "2020-01-31"


all_data = pd.read_csv('all_close_volume.csv',  header = 0 , parse_dates = True, index_col = 0)
all_data['amzn_return'] = all_data['amzn_close'] / all_data['amzn_close'].shift() - 1
all_data['fb_return'] = all_data['fb_close'] / all_data['fb_close'].shift() - 1
all_data['intc_return'] = all_data['intc_close'] / all_data['intc_close'].shift() - 1

all_data['amzn_label'] = np.where(all_data['amzn_return'] > 0, 1, 0)
all_data['fb_label'] = np.where(all_data['fb_return'] > 0, 1, 0)
all_data['intc_label'] = np.where(all_data['intc_return'] > 0, 1, 0)

all_data['amzn_return'].plot(kind = 'hist')
all_data["amzn_std_return"].plot(kind = 'hist')

all_data["amzn_std_return"] = (all_data['amzn_return'] - all_data['amzn_return'][:val_start_date].mean()) / all_data['amzn_return'][:val_start_date].std()
all_data["amzn_std_volume"] = (all_data["amzn_volume"] - all_data["amzn_volume"].rolling(50).mean()) / all_data["amzn_volume"].rolling(50).std()
all_data["fb_std_return"] = (all_data['fb_return'] - all_data['fb_return'][:val_start_date].mean()) / all_data['fb_return'][:val_start_date].std()
all_data["fb_std_volume"] = (all_data["fb_volume"] - all_data["fb_volume"].rolling(50).mean()) / all_data["fb_volume"].rolling(50).std()
all_data["intc_std_return"] = (all_data['intc_return'] - all_data['intc_return'][:val_start_date].mean()) / all_data['intc_return'][:val_start_date].std()
all_data["intc_std_volume"] = (all_data["intc_volume"] - all_data["intc_volume"].rolling(50).mean()) / all_data["intc_volume"].rolling(50).std()


from statsmodels.tsa.stattools import adfuller

adfuller(all_data.amzn_std_return)

all_data.dropna(inplace = True)

all_data.dropna(inplace = True)
all_data.head()


val_start_iloc  = all_data.index.get_loc(val_start_date,  method = 'bfill')
test_start_iloc = all_data.index.get_loc(test_start_date, method = 'bfill' )


train_generator = TimeseriesGenerator(all_data[["amzn_std_return", "amzn_std_volume", "fb_std_return", "fb_std_volume", "intc_std_return", "intc_std_volume"]].values,
                                      all_data[["intc_label"]].values,
                                      length = 7, batch_size = 64,
                                      end_index = val_start_iloc-1)
val_generator   = TimeseriesGenerator(all_data[["amzn_std_return", "amzn_std_volume", "fb_std_return", "fb_std_volume", "intc_std_return", "intc_std_volume"]].values, all_data[["intc_label"]].values,
                                    length = 7, batch_size = 64, start_index = val_start_iloc,
                                    end_index = test_start_iloc-1)
test_generator = TimeseriesGenerator(all_data[["amzn_std_return", "amzn_std_volume", "fb_std_return", "fb_std_volume", "intc_std_return", "intc_std_volume"]].values, all_data[["intc_label"]].values,
                                     length = 7, batch_size = 64, start_index = test_start_iloc)


def model_fn(params):
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(params["lstm_size"], input_shape = (7, 6))) 
    model.add(tf.keras.layers.Dropout(params["dropout"])) #  regularisation
    model.add(tf.keras.layers.Dropout(params["dropout"]))
    model.add(tf.keras.layers.Dense(1, activation = "sigmoid"))
    
    model.compile(optimizer = tf.keras.optimizers.Adam(params["learning_rate"]),
                  loss = "binary_crossentropy", metrics = ["accuracy"])            

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor = "val_accuracy", patience = 5,
                                                  restore_best_weights = True)]
    history   = model.fit_generator(train_generator, validation_data = val_generator,
                                  callbacks = callbacks, epochs = 100, verbose = 0).history
    
    return (history, model)


def random_search(model_fn, search_space, n_iter, search_dir):
    
    results = [] 
    
    os.mkdir(search_dir) 
    
    best_model_path = os.path.join(search_dir, "best_model.h5")
    results_path    = os.path.join(search_dir, "results.csv")
    
    for i in range(n_iter):
        
        params           = {k: v[np.random.randint(len(v))] for k, v in search_space.items()}
        history, model   = model_fn(params)
        epochs           = np.argmax(history["val_accuracy"]) + 1
        result           = {k: v[epochs - 1] for k, v in history.items()}
        params["epochs"] = epochs
        
        if i == 0:
            
            best_val_accuracy = result["val_accuracy"]
            model.save(best_model_path)
            
        if result["val_accuracy"] > best_val_accuracy:
            best_val_accuracy = result["val_accuracy"]
            model.save(best_model_path)
            
        result = {**params, **result}
        results.append(result)
        tf.keras.backend.clear_session()
        print(f"iteration {i + 1} – {', '.join(f'{k}:{v:.4g}' for k, v in result.items())}")
        
    best_model = tf.keras.models.load_model(best_model_path)
    results    = pd.DataFrame(results)
    
    results.to_csv(results_path)
    
    return (results, best_model)

search_space = {"lstm_size":     np.linspace(50, 200, 3, dtype = int),
                "dropout":       np.linspace(0, 0.4, 2),
                "learning_rate": np.linspace(0.004, 0.01, 5)}


iterations          = 20
results, best_model = random_search(model_fn, search_space, iterations, "search_new")
results.sort_values("val_accuracy", ascending = False).head()

best_model.evaluate_generator(test_generator)
heatmap_data=all_data[['amzn_std_return','amzn_std_volume','fb_std_return','fb_std_volume','intc_std_return','intc_std_volume','intc_label']].copy()

heatmap=sns.heatmap(heatmap_data.corr()[['intc_label']].sort_values(by='intc_label', ascending=False), vmin=-1, vmax=1, annot=False)
heatmap.set_title('Correlation of different variables to Intel price movement', fontdict={'fontsize':18}, pad=16)
