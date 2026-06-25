This module is responsible for the generation of ad opportunity feature and value prediction with our trained model, corresponding to Section 3.1 in the AuctionNet paper.

# File Structure

File structure under `model_utils`:

```
model_utils
|── data                         # Generated traffic data and statistical data on traffic distribution over time.
|── check_point                  # Model parameters for the traffic prediction model, statistical measures for data processing, etc.
|── stats_useful                 # Auxiliary data used for transforming traffic data, such as the meanings of various features.
|── build_data.py                # Converts traffic data in CSV format into vectors, which serve as inputs for the model predicting traffic value.
|── model_utils.py               # Module for building the model and other utility functions. 
|── PV_pred.py                   # Uses the model to predict traffic value and further processes the data to obtain traffic and value data.
|── PvModel.py                   # Model for predicting traffic value.
```



