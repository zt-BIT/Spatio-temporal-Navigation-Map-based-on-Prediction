# Spatio-temporal-Navigation-Map-based-on-Prediction
This code is for the work of 《Trajectory Prediction-based Local Spatio-temporal Navigation Map for Autonomous Driving in Dynamic Highway Environment》.

## Introduction

In this work, we construct a framework of **LSTM (Long-short Term Memory)+MDN** to predict the possible future trajectories of surrounding vehicles in 3s. Then, the prediction results are projected into the coordinate system whose origin is from the perspective of the ego vehicle, and a 3D spatio-temporal map is constructed. Through the spatio-temporal map, the dynamic obstacles around the ego vehicle are transformed into static obstacles extending along the time axis, thus the dynamic obstacle avoidance problem is tranlated into static obstacle avoidance problem, assisting the ego vehicle in decision-making and local path planning.

## Prerequisites

* tensorflow 1.8.0
* sklearn 0.21.3
* joblib 0.13.2
* numpy 1.16.0
* pandas 0.23.1
* hyperopt 0.1.2 

## Files

* **main.py**: main funtion. To run: python main.py --train_eval  
* **models**: model OBJECT  
* **utils**: 

  * dataloader.py    
  * util_MDN.py: calculate the pdf given Gaussian params  
  * get_tensor.py: fetch tensor from the graph  
* **EXAMPLE_DATA**: example testing data  
* **SAVED_MODEL**: trained LSTM_MDN model

## Prediction Results

* Scenario1   

![](https://github.com/zt600158/Spatio-temporal-Navigation-Map-based-on-Prediction/blob/master/figs/scenario1.png)

* Scenario2   

![](https://github.com/zt600158/Spatio-temporal-Navigation-Map-based-on-Prediction/blob/master/figs/scenario2.png)

* Scenario3

![](https://github.com/zt600158/Spatio-temporal-Navigation-Map-based-on-Prediction/blob/master/figs/scenario3.png)

## Spatio-temporal Navigation Map

* lane keeping  

 * 3D map  
 ![](https://github.com/zt600158/Spatio-temporal-Navigation-Map-based-on-Prediction/blob/master/figs/lane_keep.png)
 
 * top view & * cross-section of t=1.0s & * cross-section of t=2.0s  

 <div align=center>
    <img src="https://github.com/zt600158/Spatio-temporal-Navigation-Map-based-on-Prediction/blob/master/figs/top_view_keep.png" width="200"/><img src="https://github.com/zt600158/Spatio-temporal-Navigation-Map-based-on-Prediction/blob/master/figs/lane_keep_t10.png" width="200"/><img src="https://github.com/zt600158/Spatio-temporal-Navigation-Map-based-on-Prediction/blob/master/figs/lane_keep_t20.png" width="200"/>
</div>

* lane changing  

 * 3D map  
 ![](https://github.com/zt600158/Spatio-temporal-Navigation-Map-based-on-Prediction/blob/master/figs/lane_change.png)
 
 * top view & * cross-section of t=1.0s & * cross-section of t=2.0s  

 <div align=center>
    <img src="https://github.com/zt600158/Spatio-temporal-Navigation-Map-based-on-Prediction/blob/master/figs/top_view_change.png" width="200"/><img src="https://github.com/zt600158/Spatio-temporal-Navigation-Map-based-on-Prediction/blob/master/figs/lane_change_t10.png" width="200"/><img src="https://github.com/zt600158/Spatio-temporal-Navigation-Map-based-on-Prediction/blob/master/figs/lane_change_t20.png" width="200"/>
</div>
