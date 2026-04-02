#Plotting imports
import hvplot.pandas  # import hvplot extension for pandas
import hvplot
import holoviews as hv
import matplotlib.colors as mcolors

from pandas import DataFrame
import numpy as np
import os

from copy import deepcopy
import random
from .models import myModel


def PartialDependencePlot(model_Pipeline,x_test,batch_size=2**9,ins_cho_indx=[1,2],ins_cho_max=[29,270],
                          sample_size=300,save_to="./",verbose=0,PH_idx=0,impulse_timepoint=-1):
    """
    Computes and plots partial dependence-style responses for a given model.

    This function analyzes the model's response to varying insulin and carbohydrate inputs. 
    It works by perturbing real data points with different impulse values for insulin and CHO, 
    predicting their outcomes, and comparing them against a baseline prediction. The results, 
    summarized as median and interquartile range (IQR), are then plotted and saved.

    The main steps are:
    1. Sample a subset of `x_test`.
    2. Generate synthetic variations by applying a range of impulse amplitudes for insulin and CHO.
    3. Predict the model's output for these variations.
    4. Calculate the change in prediction (Δ-prediction) relative to a baseline (i.e., without  any impulse).
    5. Aggregate the results to compute median and IQR.
    6. Generate and save plots if a path is provided.

    Parameters
    ----------
    model_Pipeline : object
        A trained model pipeline that has a `predict()` method.
    x_test : sequence-like
        A list or array of test examples, where each example is a time-series array.
    batch_size : int, optional
        The batch size for model predictions (default is 2**9).
    ins_cho_indx : list or tuple of two ints, optional
        The indices for the insulin and CHO features in the input data (default is [1, 2]).
    ins_cho_max : list or tuple of two ints, optional
        The maximum amplitudes for insulin and CHO impulses (default is [29, 270]).
    sample_size : int, optional
        The number of examples to sample from `x_test` (default is 300).
    save_to : str or None, optional
        The file path to save the output plots. If None, no plots are saved (default is "./").
    verbose : int, optional
        The verbosity level for model predictions (default is 0).
    PH_idx : int, optional
        The index of the prediction horizon to use from the model's output (default is 0).
    impulse_timepoint : int, optional
        The time index at which to apply the impulse (default is -1).

    Returns
    -------
    pred_ins : numpy.ndarray
        The median Δ-prediction values for each insulin amplitude.
    pred_CHO : numpy.ndarray
        The median Δ-prediction values for each CHO amplitude.
    layout : object
        A Holoviews layout object containing the generated plots.

    Notes
    -----
    - The function calculates the Δ-response by subtracting a per-sample baseline prediction where the target input (insulin or CHO) is zeroed out.
    - The final plots for CHO and insulin responses are saved as an HTML file if `save_to` is specified.
    """
    # Ensure sample size does not exceed the number of available test examples
    sample_size = min(sample_size, len(x_test)) 
    # Randomly sample examples from the test set without replacement
    random_idx = random.sample(range(len(x_test)), sample_size) 
    X = deepcopy([x_test[i] for i in random_idx])
    # Define the range of impulse values for insulin and CHO
    ins_range=range(0,ins_cho_max[0]+1)
    CHO_range=range(0,ins_cho_max[1]+1)
    # Initialize arrays to store prediction results
    pred_ins=np.zeros((len(ins_range),1))
    prc_ins = np.zeros((len(ins_range),2)) # For 25th and 75th percentiles
    pred_CHO=np.zeros((len(CHO_range),1))
    prc_CHO = np.zeros((len(CHO_range),2)) # For 25th and 75th percentiles

    # Create baseline observations for comparison
    X_np = np.array(X)  # shape: (num_samples, timesteps, features)
    baseline_CHO=X_np.copy() #baseline CHO
    baseline_CHO[:, :, ins_cho_indx[1]] = 0 # set the cho to 0
    baseline_ins=X_np.copy() #baseline insulin
    baseline_ins[:, :, ins_cho_indx[0]] = 0 # set the insulin to 0
    observations = []

    # Generate perturbed observations for each impulse value
    for J in ins_range:
        X_ins = baseline_ins.copy()
        X_ins[:, impulse_timepoint, ins_cho_indx[0]] = J # Apply insulin impulse
        observations.append(X_ins)


    for J in CHO_range:
        X_cho = baseline_CHO.copy()
        X_cho[:, impulse_timepoint, ins_cho_indx[1]] = J # Apply CHO impulse
        observations.append(X_cho)
    
    # Unstack observations for batch prediction
    observations=[x for obs in observations for x in obs] #for each 3D array, unstack it into 2D arrays

    # Perform predictions on all observations in parallel
    predictions=model_Pipeline.predict(observations,batch_size=batch_size,verbose=verbose)[:,PH_idx] 
    baseline_pred_ins=model_Pipeline.predict([x for x in baseline_ins],batch_size=batch_size,verbose=verbose)[:,PH_idx] 
    baseline_pred_CHO=model_Pipeline.predict([x for x in baseline_CHO],batch_size=batch_size,verbose=verbose)[:,PH_idx] 
    # Resize baseline predictions for broadcasting
    baseline_pred_ins=np.resize(baseline_pred_ins, (len(ins_range)*sample_size,)) 
    baseline_pred_CHO=np.resize(baseline_pred_CHO, (len(CHO_range)*sample_size,)) 
    # Subtract baseline predictions to get the Δ-prediction
    predictions = predictions - np.concatenate([baseline_pred_ins, baseline_pred_CHO])


    # Aggregate results to compute median and percentiles
    counter=0
    for J in ins_range:
        pred_ins[J] = np.median(predictions[counter*sample_size:(counter+1)*sample_size])
        prc_ins[J,:] = np.percentile(predictions[counter*sample_size:(counter+1)*sample_size],[25,75])
        counter+=1

            
    for J in CHO_range:
        pred_CHO[J] = np.median(predictions[counter*sample_size:(counter+1)*sample_size])
        prc_CHO[J,:] = np.percentile(predictions[counter*sample_size:(counter+1)*sample_size],[25,75])
        counter+=1

    # Generate and save plots if a path is provided
    if save_to is not None:

        # Create a DataFrame for CHO results
        df_CHO=DataFrame(data={'median':pred_CHO.flatten(),'CHO':CHO_range})
        df_CHO['25th percentile'] = prc_CHO[:,0]
        df_CHO['75th percentile'] = prc_CHO[:,1]

        # Create the plot for CHO response
        plot_CHO = df_CHO.hvplot.line(x='CHO',y='median', xlabel="CHO [g]", ylabel="ΔBG [mg/dL]", 
                                      width=500, height=400,line_width=3.0,legend="right",label="Median").opts(show_grid=True)
        plot_CHO *= df_CHO.hvplot.area(x='CHO',y='25th percentile',y2='75th percentile',fill_alpha=0.2, 
                                       line_alpha=0,hover=False,legend="right")
        plot_CHO *= df_CHO.hvplot.line(x='CHO',y=['25th percentile','75th percentile'], 
                                        line_dash='dashed', line_color='black', line_width=1.5,legend="right")
        

        # Create a DataFrame for insulin results
        df_Ins=DataFrame(data={'median':pred_ins.flatten(), 'Ins':ins_range})
        df_Ins['25th percentile'] = prc_ins[:,0]
        df_Ins['75th percentile'] = prc_ins[:,1]
        # Create the plot for insulin response
        plot_ins = df_Ins.hvplot.line(x='Ins',y='median', xlabel="Insulin [U]", ylabel="ΔBG [mg/dL]",
                                      width=500, height=400, line_width=3.0,legend="right",label="Median").opts(show_grid=True)
        plot_ins *= df_Ins.hvplot.area(x='Ins',y='25th percentile',y2='75th percentile',fill_alpha=0.2, 
                                       line_alpha=0,hover=False,legend="right")
        plot_ins *= df_Ins.hvplot.line(x='Ins',y=['25th percentile','75th percentile'], 
                                        line_dash='dashed', line_color='black', line_width=1.5,legend="right")

        # Combine plots into a single layout
        layout = (plot_CHO + plot_ins )

        # Set the file extension and save the plot
        _, ext = os.path.splitext(save_to)
        if ext == "":
            save_to += ".html"
        # Save to file
        hv.save(layout, save_to)
        
    return pred_ins, pred_CHO,layout

def impulse_response(model_Pipeline, impulses, save_to="./",verbose=0,Ts=5,ins_cho_indx=(1,2),
                       CGM_idx=0, impulse_timepoint=-1):
    """
    Computes and plots the model's impulse response to insulin and carbohydrate inputs.

    This function evaluates the model's response to impulses of insulin and 
    carbohydrates. It creates input scenarios with an impulse at a specific timepoint 
    and predicts the model's output. The response is then compared to a baseline 
    (with zero exogenous inputs) to calculate the change in CGM (ΔCGM). The resulting 
    responses are plotted and saved as an HTML file.

    Parameters
    ----------
    model_Pipeline : object
        A fitted model pipeline with `get_model()` and `predict()` methods.
    impulses : dict
        A dictionary containing arrays of impulse amplitudes for insulin and carbohydrates.
        The keys should be 'Insulin' and 'CHO' and the values should be lists or arrays of amplitudes.
    ins_cho_indx : tuple[int, int], optional
        The indices for the insulin and CHO features (default is (1, 2)).
    save_to : str, optional
        The file path to save the output plot. If None, no plot is saved (default is "./").
    verbose : int, optional
        The verbosity level for model predictions (default is 0).
    Ts : int, optional
        The sampling period in minutes, used for the time axis in the plots (default is 5).
    CGM_idx : int, optional
        The index of the CGM feature in the input data (default is 0).
    impulse_timepoint : int, optional
        The time index where the impulse is applied (default is -1).

    Returns
    -------
    layout : object
        A Holoviews layout object containing the generated plots.

    Notes
    -----
    - The function expects `model_Pipeline.predict` to return a 2D array of shape (n_examples, n_timepoints).
    - The baseline prediction (response to zero exogenous inputs) is subtracted from all other responses.
    - A custom colormap is used for plotting if there are many amplitudes; otherwise, a legend is displayed.
    """
    # Get the shape of the input tensor from the model
    if isinstance(model_Pipeline.get_model(), myModel): #for transformer models
        shape = model_Pipeline.get_model().input_shape[0][1:] 
    else:
        shape = model_Pipeline.get_model().input_shape[1:] #for other models
    
    # Initialize the synthetic input with average CGM
    X = np.zeros(shape) 
    X[:,CGM_idx]=145 # [mg/dL] set all the CGM series to average CGM

    # Determine the length of the output sequence
    len_output=len(model_Pipeline.predict([X],verbose=verbose).flatten())+1 # +1 to include the initial timepoint
    time=np.arange(0,len_output)*Ts
    # Create DataFrames to store results
    
    df_CHO=np.zeros((len_output,len(impulses["CHO"])))
    df_CHO=DataFrame(df_CHO,columns=[str(amp) + " g" for amp in impulses["CHO"]])
    df_CHO["time"]=time
    df_CHO.set_index("time",inplace=True)

    df_Ins=np.zeros((len_output,len(impulses["Insulin"])))
    df_Ins=DataFrame(df_Ins,columns=[str(amp) + " U" for amp in impulses["Insulin"]])
    df_Ins["time"]=time
    df_Ins.set_index("time",inplace=True)

    # Generate input scenarios for insulin and CHO impulses
    inputs=[]
    #Insulin
    for amp in df_Ins.columns:
        X[:,ins_cho_indx[0]]=0 #reset X
        X[impulse_timepoint,ins_cho_indx[0]]=float(amp.split()[0])
        inputs.append(X.copy())  	
    
    #CHO
    X[:,ins_cho_indx[0]]=0 #reset X
    for amp in df_CHO.columns:
        X[:,ins_cho_indx[1]]=0 #reset X
        X[impulse_timepoint,ins_cho_indx[1]]=float(amp.split()[0])
        inputs.append(X.copy())
    
    #predict the impulsive response for all examples
    preds=model_Pipeline.predict(inputs,verbose=verbose) #shape: (len(impulses["CHO"]) + len(impulses["Insulin"]), n_timepoints)
    preds=preds-preds[0,:] #subtract the baseline prediction (where exogenous inputs are 0)

    #assign the predictions to the dataframes
    df_Ins.iloc[1:,0:len(impulses['Insulin'])]=preds[:len(impulses['Insulin'])].transpose()
    df_CHO.iloc[1:,0:len(impulses['CHO'])]=preds[len(impulses['Insulin']):].transpose()


    # Generate plots for the impulse responses
    plot_CHO=df_CHO.hvplot.line(x="time",y=df_CHO.columns,xlabel="time [min]", ylabel="ΔBG [mg/dL]",
                                title="Model Response to Carbohydrates",width=450, height=400,
                                fontsize={'title': '10pt', 'legend_title': '10pt',
                                            'legend': '10pt','ylabel': '10pt','xlabel': '10pt','xticks': '10pt','yticks': '10pt'},
                                group_label="CHO intake",
                                ).opts(show_grid=True)
    
    plot_CHO *= df_CHO.hvplot.scatter(x="time",y=df_CHO.columns,fill_color=None, line_width=2,  )
    
    plot_Ins=df_Ins.hvplot.line(x="time",y=df_Ins.columns,xlabel="time [min]", ylabel="ΔBG [mg/dL]",
                                title="Model Response to Insulin",width=450, height=400,
                                fontsize={'title': '10pt', 'legend_title': '10pt',
                                            'legend': '10pt','ylabel': '10pt','xlabel': '10pt','xticks': '10pt','yticks': '10pt'},
                                group_label="Insulin bolus",
                                ).opts(show_grid=True)
    plot_Ins *= df_Ins.hvplot.scatter(x="time",y=df_Ins.columns,fill_color=None, line_width=2)
    
    # Combine all plots into one layout
    layout = (plot_CHO + plot_Ins)
        
    # save the plot
    _, ext = os.path.splitext(save_to)
    if ext == "":
        save_to += ".html"
    hv.save(layout, save_to)
    return layout
