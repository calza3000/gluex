#imports

#keras imports
from airt.keras.layers import MonoDense
import keras
from keras_nlp.layers import SinePositionEncoding, TransformerEncoder, TransformerDecoder
from keras.layers import Dense, Input, Conv1D, Add, BatchNormalization
import tensorflow as tf
from keras.engine import data_adapter
from keras.utils import Progbar

#utility imports
import pickle
import os
import numpy as np
from copy import deepcopy
from scipy import sparse

#sklearn imports
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import _array_api
from sklearn.utils._array_api import get_namespace
from sklearn.utils._array_api import get_namespace


#utility functions
def _handle_zeros_in_scale(scale, copy=True, constant_mask=None): #copied from sklearn.preprocessing._data.py
    """Set scales of near constant features to 1.

    The goal is to avoid division by very small or zero values.

    Near constant features are detected automatically by identifying
    scales close to machine precision unless they are precomputed by
    the caller and passed with the `constant_mask` kwarg.

    Typically for standard scaling, the scales are the standard
    deviation while near constant features are better detected on the
    computed variances which are closer to machine precision by
    construction.
    """
    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == 0.0:
            scale = 1.0
        return scale
    # scale is an array
    else:
        xp, _ = get_namespace(scale)
        if constant_mask is None:
            # Detect near constant values to avoid dividing by a very small
            # value that could lead to surprising results and numerical
            # stability issues.
            constant_mask = scale < 10 * xp.finfo(scale.dtype).eps

        if copy:
            # New array to avoid side-effects
            scale = xp.asarray(scale, copy=True)
        scale[constant_mask] = 1.0
        return scale
    
def load_transformer(path,model_fun):
    """
    Loads a trained Transformer model from a specified path.

    This function restores a Keras-based Transformer model by first loading the 
    full model from the given path and then transferring its weights to a new 
    instance created by `model_fun`.

    Parameters:
    -----------
    path : str
        The file path to the saved model.
    model_fun : function
        A function that returns a new, un-trained instance of the Transformer model.

    Returns:
    --------
    model : object
        The trained Transformer model with weights loaded.

    Example:
    --------
    >>> model = load_transformer("Models/transformer.tf", Transformer_Net)
    >>> predictions = model.predict(x_test)
    """
    keras_model= keras.models.load_model(path)
    model=model_fun()
    model.set_weights(keras_model.get_weights())
    return model

def Transformer_Net(input_shape=(18,3), output_shape=(18,), dropout_rate=0.15, kernel_size=3,
                    encoder_layer=3, decoder_layer=3, intermediate_dim=128, d_model=32, num_heads=8,cnn_layers=1):
    """
    Constructs a Transformer-based neural network for sequence-to-sequence tasks.

    This function builds a model with a CNN feature extractor followed by a Transformer
    encoder-decoder architecture. It is designed for time-series forecasting but can be
    adapted for other sequence tasks.

    Parameters:
    -----------
    input_shape : tuple, optional
        The shape of the input data (timesteps, features), default is (18, 3).
    output_shape : tuple, optional
        The shape of the output sequence, default is (18,).
    dropout_rate : float, optional
        The dropout rate for regularization, default is 0.15.
    kernel_size : int, optional
        The kernel size for the convolutional layers, default is 3.
    encoder_layer : int, optional
        The number of layers in the Transformer encoder, default is 3.
    decoder_layer : int, optional
        The number of layers in the Transformer decoder, default is 3.
    intermediate_dim : int, optional
        The dimension of the feed-forward network in the Transformer blocks, default is 128.
    d_model : int, optional
        The dimensionality of the model's embeddings, default is 32.
    num_heads : int, optional
        The number of attention heads in the multi-head attention layers, default is 8.
    cnn_layers : int, optional
        The number of initial convolutional layers for feature extraction, default is 1.

    Returns:
    --------
    myModel
        A custom Keras model (`myModel`) with the specified Transformer architecture.
    """
    input_layer = Input(input_shape)

    # The 'forced_output' is used for teacher forcing during training.
    forced_output = Input(output_shape)
    shifted_output = ShiftOutputLayer()(forced_output)

    # Masking layer to ignore padded values (25) in the input sequences.
    encoder = keras.layers.Masking(mask_value=25)(input_layer)
    decoder = keras.layers.Masking(mask_value=25)(shifted_output)

    # Temporal feature extraction using 1D convolutions.
    for _ in range(cnn_layers):
        encoder = Conv1D(filters=d_model, kernel_size=kernel_size,
                     strides=1, padding="same")(encoder)
        encoder = keras.layers.PReLU()(encoder)
        encoder = BatchNormalization()(encoder)

    for _ in range(cnn_layers):
        decoder = Conv1D(filters=d_model, kernel_size=kernel_size, 
                     strides=1, padding="causal")(decoder)  # "causal" padding ensures that output at time t does not depend on input at time t+1.
        decoder = keras.layers.PReLU()(decoder)
        decoder = BatchNormalization()(decoder)

    # Add positional encodings to the encoder and decoder inputs.
    encoder = Add()([encoder, SinePositionEncoding()(encoder)])
    decoder = Add()([decoder, SinePositionEncoding()(decoder)])

    # Transformer Encoder stack.
    for i in range(encoder_layer):
        encoder = TransformerEncoder(
            intermediate_dim=intermediate_dim, num_heads=num_heads, dropout=dropout_rate)(encoder)

    # Transformer Decoder stack.
    for i in range(decoder_layer):
        decoder = TransformerDecoder(intermediate_dim=intermediate_dim,
                                     num_heads=num_heads, dropout=dropout_rate)(decoder, encoder)

    # Final output layer with a time-wise dense transformation.
    output_layer = Dense(1, 'sigmoid')(decoder)
    output_layer = keras.layers.Reshape((-1,))(output_layer)

    return myModel(seq_length=output_shape[0], inputs=(input_layer, forced_output), outputs=output_layer)

def load_model(model_path):
    """
    Loads a Keras model and its associated preprocessing pipeline from a specified path.

    This function handles loading both standard Keras models (.keras) and custom 
    Transformer models (.tf). It also loads the corresponding fitted scaler objects 
    for data preprocessing.

    Parameters:
    -----------
    model_path : str
        The file path to the saved model.

    Returns:
    --------
    AlreadyFittedWrap
        A wrapper object that bundles the model and its preprocessing scalers.
    """
    # Load the model based on its file extension.
    if model_path.endswith('.tf'):
        model = load_transformer(model_path,Transformer_Net)
    else:
        custom_objects={'MonoDense':MonoDense} if "PhyNet" in model_path else None
        model = keras.models.load_model(model_path,custom_objects=custom_objects)

    # Initialize the preprocessing pipeline wrapper.
    pip = AlreadyFittedWrap(model,scaler=SequencePreprocess(),output_scaler=allMinMaxScaler())

    folder_name = os.path.dirname(model_path)
    # Load the fitted scalers from disk.
    pip.scaler = pickle.load(open(os.path.join(folder_name, 'scaler.pkl'), 'rb'))  
    pip.output_scaler = pickle.load(open(os.path.join(folder_name, 'output_scaler.pkl'), 'rb'))  

    return pip

def create_mock_data(num_examples, seq_length, PH):
    """Creates mock sequence data for testing and demonstration purposes.

    This function generates synthetic input and output data that mimics the
    structure of multivariate blood glucose data (blood glucose [mg/dL],
    insulin [U], and carbohydrate intake [g]).

    The data generation process is as follows:
    - Blood Glucose (BG): Each sequence starts with a random baseline BG
      value between 50 and 300 mg/dL. A random walk is then simulated by
      adding cumulative random noise (from a normal distribution) at each
      time step to mimic physiological fluctuations.
    - Insulin: Insulin administrations are modeled as sparse events.
      Assuming 4 injections per day, specific time points are randomly
      selected to receive an insulin dose. The dose amounts are drawn from a
      log-normal distribution to reflect realistic dosing patterns.
    - Carbohydrates (CHO): Similar to insulin, carbohydrate intake (meals)
      is modeled as sparse events, assuming 4 meals per day. The size of
      each meal (in grams) is also drawn from a log-normal distribution.
    - Output BG: The target BG sequence for the prediction horizon is
      generated by continuing the random walk from the last BG value of the
      input sequence.

    Note:
        This process generates a synthetic dataset that does not fully capture
        the complex dynamics of real-world glucose, insulin, and carbohydrate
        data, but it is sufficient for demonstrating the tools in this package.

    Parameters
    ----------
    num_examples : int
        The number of examples to generate.
    seq_length : int
        The length of each input sequence (in timesteps).
    PH : int
        The length of the output sequence (prediction horizon in timesteps).

    Returns
    -------
    x_data : list
        A list of NumPy arrays representing the input sequences. Each array has
        a shape of `(seq_length, 3)`.
    y_data : np.ndarray
        A NumPy array of shape `(num_examples, PH)` representing the output
        sequences.
    """
    # Initialize input data array.
    x_data = np.zeros((num_examples, seq_length, 3), dtype="float32")

    # --- Blood Glucose (BG) Simulation in mg/dL ---
    # Simulate a physiological random walk for blood glucose.
    # Start with a baseline glucose level for each example.
    bg_baseline = np.random.uniform(40, 200, size=(num_examples, 1))
    bg_walk = np.cumsum(np.random.normal(loc=0, scale=3, size=(num_examples, seq_length)), axis=1)
    x_data[:, :, 0] = bg_baseline + bg_walk

    # --- Insulin Simulation in Units (U) ---
    # Define a helper to get sparse indices for events.
    def sparse_idx(num_examples, seq_length, freq):
        n_events = int(num_examples * seq_length * freq)
        rows = np.random.randint(0, num_examples, size=n_events)
        cols = np.random.randint(0, seq_length, size=n_events)
        return rows, cols

    # Assume 4 insulin injections per day (288 samples/day at 5-min intervals).
    freq_ins = 4 / 288
    ins_rows, ins_cols = sparse_idx(num_examples, seq_length, freq_ins)
    # Generate insulin doses from a log-normal distribution for more realistic values.
    x_data[ins_rows, ins_cols, 1] = np.random.lognormal(mean=1.5, sigma=0.5, size=len(ins_rows))

    # --- Carbohydrate (CHO) Simulation in grams (g) ---
    # Assume 4 meals per day.
    freq_cho = 4 / 288
    cho_rows, cho_cols = sparse_idx(num_examples, seq_length, freq_cho)
    # Generate meal sizes from a log-normal distribution.
    x_data[cho_rows, cho_cols, 2] = np.random.lognormal(mean=3.5, sigma=0.4, size=len(cho_rows))

    # --- Output BG Simulation ---
    # The output sequence should realistically follow the input sequence.
    # Start the output from the last BG value of the input sequence.
    output_baseline = x_data[:, -1, 0].reshape(-1, 1)
    # Continue the random walk for the prediction horizon.
    y_data = output_baseline + np.cumsum(np.random.normal(loc=0, scale=3, size=(num_examples, PH)), axis=1)

    # Ensure data types are correct.
    x_data = x_data.astype("float32")

    # Convert to a list of arrays for compatibility with certain preprocessors.
    x_data = [x for x in x_data]
    return x_data, y_data

class SequencePreprocess(TransformerMixin):
    """
    A preprocessing transformer for scaling and padding sequence data.

    This class applies a specified scaler to each sequence in a list and then pads 
    them to a uniform length. It is designed to work within a scikit-learn pipeline.

    Parameters:
    -----------
    scaler : object, optional
        A scikit-learn-compatible scaler class (e.g., MinMaxScaler). If None, 
        MinMaxScaler is used by default.
    **kwargs : dict
        Additional keyword arguments to pass to the scaler's constructor.

    Example:
    --------
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> preprocessor = SequencePreprocess(scaler=MinMaxScaler)
    >>> preprocessor.fit(x_train)
    >>> x_transformed = preprocessor.transform(x_test)
    """

    def __init__(self, scaler=None, **kwargs):
        self.isfitted = False
        if scaler is None:
            self._scaler = MinMaxScaler(**kwargs)
        else:
            self._scaler = scaler(**kwargs)

    def fit(self, X:list,y_train=None, **kwargs):
        """
        Fits the scaler to the provided sequence data.

        The sequences in the input list are concatenated to fit the scaler on the 
        entire dataset.

        Parameters:
        -----------
        X : list
            A list of input sequences, where each sequence is a NumPy array.
        y_train : None
            This parameter is not used and is included for compatibility.
        **kwargs : dict
            Additional keyword arguments for the scaler's fit method.

        Returns:
        --------
        self : object
            The fitted SequencePreprocess instance.
        """
        if self.isfitted:
            return self
        if isinstance(X, list):
            # Concatenate all sequences to fit the scaler on the full data distribution.
            X = np.concatenate(X,dtype="float32")
        
        self._scaler.fit(X, **kwargs)
        self.isfitted = True
        return self

    def transform(self, X:list,y_train=None,pad_val=25, **kwargs):
        """
        Transforms and pads the input sequences.

        Each sequence is first scaled, then masked for NaN values, and finally padded 
        to ensure all sequences have the same length.

        Parameters:
        -----------
        X : list
            A list of input sequences to be transformed.
        y_train : None
            This parameter is not used and is included for compatibility.
        pad_val : int or float, optional
            The value used for padding, default is 25.
        **kwargs : dict
            Additional keyword arguments for the scaler's transform method.

        Returns:
        --------
        X_transformed : np.ndarray
            A single NumPy array containing the transformed and padded sequences.
        """
        if isinstance(X, list):
            # Apply scaling and masking to each sequence.
            X = [self._scaler.transform(x, **kwargs) for x in X]
            X=[self._masking(x,pad_val) for x in X] # Replace NaNs with the padding value.
            # Pad sequences to the same length.
            X = keras.utils.pad_sequences(X, dtype = "float32", value = pad_val, padding = "post")
        else:
            # If input is not a list, transform it directly.
            X = self._scaler.transform(X, **kwargs)
        return X
    
    def _masking(self, x, pad_val):
        """
        Replaces rows containing NaN values with a specified padding value.

        This is a helper function to prepare data for Keras's masking layer.

        Parameters:
        -----------
        x : np.ndarray
            The input array.
        pad_val : int or float
            The value to use for replacing NaN-containing rows.
            
        Returns:
        --------
        np.ndarray
            The array with NaN rows replaced by the padding value.
        """
        x=deepcopy(x)
        # Find all rows that have at least one NaN value.
        flag=np.any(np.isnan(x),axis=1) 
        # Replace these rows with the padding value.
        x[flag,:]=pad_val 
        return x

class allMinMaxScaler(MinMaxScaler):
    """
    A custom MinMaxScaler that scales all features based on the global minimum and 
    maximum values across the entire dataset.

    This is particularly useful when all features represent the same physical quantity 
    measured at different time points.
    """
    def partial_fit(self, X, y=None):
        """
        Computes the global minimum and maximum of X for later scaling.

        This method is designed for online learning scenarios where the data is 
        processed in batches.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the min and max values.

        y : None
            Ignored.

        Returns
        -------
        self : object
            The fitted scaler.
        """
        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError(
                "Minimum of desired feature range must be smaller than maximum. Got %s."
                % str(feature_range)
            )

        if sparse.issparse(X):
            raise TypeError(
                "MinMaxScaler does not support sparse input. "
                "Consider using MaxAbsScaler instead."
            )

        xp, _ = get_namespace(X)

        first_pass = not hasattr(self, "n_samples_seen_")
        X = self._validate_data(
            X,
            reset=first_pass,
            dtype=_array_api.supported_float_dtypes(xp),
            force_all_finite="allow-nan",
        )

        # Compute the min and max over the entire dataset, not per-feature.
        data_min = _array_api._nanmin(X) 
        data_max = _array_api._nanmax(X) 

        if first_pass:
            self.n_samples_seen_ = X.shape[0]
        else:
            # Update global min and max with the new batch of data.
            data_min = xp.minimum(self.data_min_, data_min)
            data_max = xp.maximum(self.data_max_, data_max)
            self.n_samples_seen_ += X.shape[0]

        data_range = data_max - data_min
        self.scale_ = (feature_range[1] - feature_range[0]) / _handle_zeros_in_scale(
            data_range, copy=True
        )
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self

class AlreadyFittedWrap():
    """
    A wrapper for using pre-trained models with a preprocessing pipeline.

    This class simplifies the process of making predictions with a model that requires 
    specific data scaling, by bundling the model and its scalers together.

    Attributes:
    -----------
    model : object
        The pre-trained machine learning model.
    scaler : object
        A fitted scaler for input data preprocessing.
    output_scaler : object
        A fitted scaler for inverting the model's output predictions.
    
    Example:
    --------
    >>> # Load a pre-trained model
    >>> monoD = load_model(f"Models/Prep/CNN_PDP.keras")
    >>> 
    >>> # Wrap the model with its scalers
    >>> MonoDense_pip = AlreadyFittedWrap(monoD, scaler=SequencePreprocess(), output_scaler=MinMaxScaler())
    >>> MonoDense_pip.fit(x_train, y_train)
    >>> 
    >>> # Make predictions
    >>> pred_monoD = MonoDense_pip.predict(x_test)
    >>> RMSE_monoDense = root_mean_squared_error(y_test, pred_monoD)
    """
    def __init__(self,model,scaler=SequencePreprocess(),output_scaler=allMinMaxScaler()):
        self.scaler=scaler
        self.output_scaler=output_scaler
        self.model=model
    
    def fit(self,x_train,y_train):
        """Fits the input and output scalers to the training data."""
        self.scaler.fit(x_train)
        self.output_scaler.fit(y_train.reshape(-1,1))
    
    def _transform(self,x):
        """Applies the input scaler to the data."""
        return self.scaler.transform(x)
    
    def predict(self,x,**kwargs):
        """
        Generates predictions by first transforming the input and then inverting 
        the scaled output.
        """
        x=self._transform(x)
        return self.output_scaler.inverse_transform(self.model.predict(x,**kwargs)).squeeze()
    
    def __call__(self, x,**kwargs):
        """Allows the wrapper to be called like a function, returning predictions."""
        x = self._transform(x)
        return self.output_scaler.inverse_transform(self.model(x, **kwargs))

    #for compatibility with other functions
    def get_model(self):
        """Returns the underlying Keras model."""
        return self.model
    
class myModel(keras.models.Model):
    """
    A custom Keras model with an overridden `predict` method for autoregressive forecasting.

    This class is designed for sequence-to-sequence models like Transformers, where 
    predictions are generated one step at a time.

    Attributes:
    -----------
    seq_length : int
        The length of the output sequence to be generated.
    """
    def __init__(self,seq_length, *args, **kwargs):
        self.seq_length=seq_length
        super().__init__(*args, **kwargs)
        
    @tf.function(reduce_retracing=True)
    def _predict_loop(self, x, padval):
        """Full autoregressive loop compiled to a single TF graph for speed."""
        batch_size = tf.shape(x)[0]
        output_shape = self.seq_length

        # Initialize output tensor: CLS token at position 0, padval elsewhere.
        cls_col = tf.ones([batch_size, 1, 1], dtype=tf.float32)
        pad_cols = tf.fill([batch_size, output_shape - 1, 1], padval)
        output = tf.concat([cls_col, pad_cols], axis=1)

        for i in range(output_shape):
            preds = self([x, output], training=False)
            step_pred = tf.reshape(preds[:, i], [-1, 1])  # (batch, 1)

            # In-place-style update via scatter (avoids creating new tensors with concat)
            indices = tf.stack([tf.range(batch_size),tf.fill([batch_size], i)], axis=1)
            output = tf.tensor_scatter_nd_update(output, indices, step_pred)

        return output

    # Override the predict function
    def predict(self, x, padval=25, batch_size=32, **kwargs):
        """ 
        Generates predictions autoregressively, one timestep at a time.

        Parameters:
        -----------
        x : tensor or list of tensors
            The input data for the model.
        padval : int, optional
            The padding value used to initialize the output sequence, default is 25.
        batch_size : int, optional
            The number of samples per batch, default is 32.
        **kwargs : dict
            Additional keyword arguments for the underlying predict function.

        Returns:
        --------
        output : np.ndarray
            The final predicted output sequence.
        """
        x = np.asarray(x, dtype="float32")
        padval_f = tf.cast(padval, tf.float32)

        results = []
        for start in range(0, len(x), batch_size): #for each batch
            batch = tf.convert_to_tensor(x[start:start + batch_size])
            results.append(self._predict_loop(batch, padval_f))

        return tf.concat(results, axis=0).numpy().reshape(-1, self.seq_length)
        
    def test_step(self, data,padval=25):
        """
        Performs a single test step with autoregressive prediction.

        This method is designed for evaluating Transformer-based models by generating 
        predictions autoregressively and then computing the loss and metrics.

        Parameters:
        -----------
        data : tuple
            A tuple containing the input data, target values, and sample weights.
        padval : int, optional
            The padding value for initializing the output sequence, default is 25.

        Returns:
        --------
        dict
            A dictionary of computed metrics for the test step.
        """
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        x=x[0] # Extract only the input data (not the forced output)
        batch_size = tf.shape(x)[0]
        output_shape = self.seq_length
        padval=tf.constant(padval,dtype=tf.float32)
        # Initialize the output sequence with padding values.
        output = tf.fill([batch_size, output_shape, 1], padval)  # Padding

        # Set the first token of each sequence to 1 (CLS token).
        updates = tf.ones((batch_size, 1), dtype=output.dtype)  # Updates for CLS token
        indices = tf.concat([
            tf.expand_dims(tf.range(batch_size), axis=1),
            tf.fill([batch_size, 1], 0)
          ],axis=1)  # Indices for each batch
        output = tf.tensor_scatter_nd_update(output, indices, updates)

        progbar = Progbar(output_shape, unit_name='timestep',verbose=0)
        # Autoregressively generate the output sequence.
        for i in range(output_shape): #for each timestep
            predictions = self([x, output],training=False)  # Get the predictions with a forward pass

            # select the token from the seq_len dimension
            predictions = predictions[:, i]  # (batch_size, output_shape)
            predictions = tf.reshape(predictions, (-1, 1))

        
            # concatentate the pred to the output which is given to the decoder
            # as its input.
            indices = tf.concat([
                tf.expand_dims(tf.range(batch_size), axis=1),  # Batch indices
                tf.fill([batch_size, 1], i)  # Sequence indices (all set to `i`)
            ], axis=1)
            output = tf.tensor_scatter_nd_update(output, 
                                             indices=indices,
                                             updates=predictions)
            progbar.update(i + 1)  # Increment by 1 to account for 0-based indexing

        output = tf.reshape(output, (-1, output_shape))  # Reshape output tensor to (-1, output_shape)
        y_pred = output
        # Update loss and metrics.
        self.compute_loss(x, y, y_pred, sample_weight)
        return self.compute_metrics(x, y, y_pred, sample_weight)   

class ShiftOutputLayer(keras.layers.Layer):
    """
    A Keras layer that shifts the output sequence to the right by one timestep.

    This layer is typically used in the decoder of a Transformer model to ensure that 
    the prediction for a given timestep is based on the previous timesteps' outputs.

    Attributes:
    -----------
    None

    Methods:
    --------
    call(self, forced_output):
        Implements the call method of the ShiftOutputLayer.

    """  
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    
    def call(self, forced_output):
        """ 
        Performs the right-shift operation on the input tensor.

        Parameters:
        -----------
        forced_output : tensor
            The input tensor representing the target sequence.

        Returns:
        --------
        shifted_output : tensor
            The right-shifted output tensor, ready for the decoder.
        """
        # Concatenate a start token (1s) with the sequence, excluding the last element.
        shifted_output = keras.layers.Concatenate()([tf.ones((tf.shape(forced_output)[0], 1)), forced_output[:, :-1]])
        shifted_output = tf.expand_dims(shifted_output,2)
      
        return shifted_output
