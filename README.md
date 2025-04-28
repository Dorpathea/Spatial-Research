# Spatial-Research
Research done at the University of Maine under Dr. Yifeng Zhu looking to improve Long Sequence Time-Series Forecasting using a Spatial Embedding. 

This research builds on the previous transformer models [Informer](https://github.com/zhouhaoyi/Informer2020) and [Crossformer](https://github.com/Thinklab-SJTU/Crossformer) by adding a Spatial Embedding to each model. An extensive look into the research is available in the [paper](https://digitalcommons.library.umaine.edu/etd/4049/). 

The motivation behind this research was to look into relationship between multiple locations of data, allow for analysis of data with asynchronous input, see if models can improve on datasets outside common ones, and improve prediction of current state of the art models. 


## Spatial Emebedding
The Spatial Embedding was added to the models to allow the them to learn from locality in a dataset. Some of the datasets used latitude and longitude values for this embedding, while others were given an arbitrary value due to lack of positional data.

<p align="center">
<img src=".\Photos\Spatial_Embedding.jpg" height = "320" alt="" align=center />
<br><br>
<b>Figure 2.</b> How the Spatial Embedding is Implemented into Another Model's Structure.
</p>

The code for the Spatial Embedding is similar in both models, but varies slightly to be incorporated into the current model's structure. 

## Data
This research was tested on many different datasets: [Weather](https://www.ncei.noaa.gov/data/local-climatological-data/), [ECL](https://doi.org/10.24432/C58C86), [AirQuality](https://doi.org/10.24432/C5RK5G), and [Manufacturing](https://www.kaggle.com/dsv/8684322). The datasets used for both models in this research can be found in `Spatial-Research/Informer_Model/data/custom_data` with the exception of the ECL dataset. This dataset is too large and is found (FIGURE OUT WHERE TO PUT IT) instead. 

The datasets were preprocessed to contain a label vector for identification between the different locations. How the data was preprocessed can be found in Colab: [Data Preprocessing Colab](). An example of the dataset layout can be seen below.

<p align="center">
<img src=".\Photos\Dataset_Example.png" height = "320" alt="" align=center />
<br><br>
<b>Figure 2.</b> Example of a Dataset with the Label Vector.
</p>

## Usage
There were many different code files used in this research to analyze multiple potential avenues. Below explains what each of them were used for. Running any of these tests follows a similar layout to their parent models. 

An example on how to run the Spatial Crossformer code is as follows: 

    `python3 /path_to_code/spatial_crossformer_Weather.py --data 'close_combine' --root_path '/path_to_data/' --data_path 'close_combine.csv' --in_len 168 --out_len 24 --seg_len 6 --data_dim 7 --itr 1 --learning_rate 1e-5 --dropout 0.2 --train_epochs 30 --patience 40 --data_split '0.6,0.2,0.2' --n_heads 1 --e_layer 1 --d_ff 64 --d_model 64 --name 'name_of_test'`

An example on how to run the Spatial Informer code is as follows: 

    `python3 /path_to_code/spatial_informer_no_embed.py --model 'informer' --data 'spatial' --root_path '/path_to_data/' --data_path 'Spatial_ECL.csv' --features 'M' --target 'MT' --freq 'h' --seq_len 720 --label_len 336 --pred_len 720 --enc_in 2 --dec_in 2 --c_out 2 --d_model 512 --n_heads 8 --e_layers 3 --d_layers 2 --d_ff 2048 --attn 'prob' --embed 'learned' --des 'Exp' --itr 1 --factor 5 --learning_rate 1e-4 --dropout 0.05 --train_epochs 5 --patience 40 --name 'name_of_test'`


### Informer
In `Spatial-Research/Informer_Model` are the files to run the different tests. 

`main_informer.py` is the regular code to run the Informer model.

`spatial_informer.py` is the Informer model with the Spatial embedding using 2 linears and the new dataloader. 

`spatial_informer_latlon.py` is the Informer model with the Spatial embedding using 2 linears and the new dataloader for the weather datasets. 

`spatial_informer_latlon_no_embed.py` is the Informer model and the new dataloader for the weather datasets. 

`spatial_informer_no_embed.py` is the Informer model with the new dataloader. 

### Crossformer
In `Spatial-Research/Crossformer_Model` are the files to run the different tests.

`eval_crossformer.py` is the code used by the main Crossformer model to evaluate the results of the model.

`eval_crossformer_spatial.py` is the code used by the Crossformer models with the Spatial embedding to evaluate the results of the model.

`eval_crossformer_spatial_weather.py` is the code used by the Crossformer models with the Spatial embedding to evaluate the results of the model for the weather datasets.

`main_crossformer.py` is the regular code to run the Crossformer model.

`spatial_crossformer.py` is the Crossformer model with the Spatial embedding using 2 linears and the new dataloader. 

`spatial_crossformer_no_embed.py` is the Crossformer model with the new dataloader. 

`spatial_crossformer_Weather.py` is the Crossformer model with the Spatial embedding using 2 linears and the new dataloader for the weather datasets. 

`spatial_crossformer_Weather_Double_Noise.py` is the Crossformer model with the Spatial embedding using 2 linears and the new dataloader for the weather datasets. It also has noise added to the training portion of the data.  

`spatial_crossformer_NewEmbed.py` is the Crossformer model with the Spatial embedding using the structure of linear→relu→linear→relu and the new dataloader for the weather datasets. 

`spatial_crossformer_NewEmbed_Noise.py`  is the Crossformer model with the Spatial embedding using the structure of linear→relu→linear→relu and the new dataloader for the weather datasets. It also has noise added to the training portion of the data.  

`spatial_crossformer_NewEmbed_Tanh.py` is the Crossformer model with the Spatial embedding using the structure of linear→relu→linear→relu→linear→tanh and the new dataloader for the weather datasets. 

`spatial_crossformer_Weather_no_embed.py` is the Crossformer model and the new dataloader for the weather datasets. 

`spatial_crossformer_Weather_Single_Noise.py` is the Crossformer model with the Spatial embedding using 1 linear and the new dataloader for the weather datasets. It also has noise added to the training portion of the data. 
