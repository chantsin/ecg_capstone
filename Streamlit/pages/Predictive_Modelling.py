import streamlit as st
import pandas as pd
import numpy as np
import keras
import wfdb
import altair as alt

# import sys
# sys.path.append('..')
from utils import ecg_cleaning as c

################################ INTRODUCTION ##################################

st.title("ECG Prediction")

st.write("Let me show you how my model works! Pass me an age and I will tell \
         you how a person's heart of this age is doing!")

st.sidebar.markdown("# ECG Prediction")
st.sidebar.markdown(
"Given an ECG signal, my recurrent deep neural network predicts the diagnostic \
    outcome of the signal with over 80% accuracy.")


############################# PATIENT INFO FILE ################################

path = '../data/physionet.org/files/ptb-xl/1.0.3/'
metadata = pd.read_csv('Streamlit/cleaned_metadata.csv', index_col=0)

############################ USER INPUT SELECTION ##############################

# Age
age = st.selectbox(
    'Age',
    tuple(str(num) for num in range(2, 90))
)

# Sex
sex = st.selectbox(
    'Sex',
    ('Male', 'Female')
)

st.write(f'An {age} year old {sex} can have an ECG like this: ')

if sex == 'Male':
    s = 0
else:
    s = 1

# Height
# height = st.selectbox(
#     'Height (cm)',
#     tuple(str(height) for height in np.sort(metadata['height'].unique()))
# )
# st.write(f'You selected: {height} cm.')

# Weight
# weight = st.selectbox(
#     'Weight (kg)',
#     tuple(str(weight) for weight in np.sort(metadata['weight'].unique()))
# )
# st.write(f'You selected: {weight} kg.')

############################## GRAB SAMPLE ECG #################################

signal = c.grab_sample(float(age), 
                       s,  
                       path, 
                       metadata)

if signal is None:
    st.write("We do not have a signal with these parameters! Please try \
             other combinations!!")

else: 
        
################################ PLOT SIGNAL ###################################

    sampling_frequency = 100  # Sampling frequency
    sig_len = 1000 # Signal length 
    time = np.arange(0, sig_len) / sampling_frequency
    dt = 1/sampling_frequency
    signals = signal.flatten()

    samp_ecg = pd.DataFrame(
        {
            'Time (seconds)': time,
            'Original ECG': signals
        }
    )

    chart = alt.Chart(samp_ecg).mark_line(color='blue').encode(
        x=alt.X('Time (seconds)'),
        y=alt.Y('Original ECG'),
    ).interactive()

    st.write("#### Lead II of the ECG Record")
    st.altair_chart(chart, theme="streamlit", use_container_width=True)

########################### PREDICTIVE MODELLING ###############################

    # Defining model and using weights
    rnn_model_2 = keras.Sequential([
        # the intermediate recurrent layers should return full sequences
        keras.layers.GRU(16, activation='relu', return_sequences=True),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.15),

        # the last recurrent layer only returns the final output
        keras.layers.GRU(16, activation='relu', return_sequences=False),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.15),

        # output layer
        keras.layers.Dense(16, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.15),
        keras.layers.Dense(1, activation='sigmoid')],
    )

    rnn_model_2.build(input_shape=(None, 1000, 1))

    # Load weights
    rnn_model_2.load_weights('rnn_model_2_weights.h5')

    # Compile
    rnn_model_2.compile(
        # Optimizer
        optimizer=keras.optimizers.Adam(learning_rate=0.005),  # learning rate can be adjusted here
        # Loss function to minimize
        loss=keras.losses.BinaryCrossentropy(),
        # Metric used to evaluate model
        metrics=[keras.metrics.BinaryAccuracy(),
                keras.metrics.Recall()]
    )

    signal_bl = c.baseline_removal(signals, freq_start=0.1, freq_stop=1.5)

    prediction = rnn_model_2.predict(
        signal_bl.reshape(-1, 1000, 1).astype("float"))
    binary_array = (prediction > 0.5).astype(int).reshape(-1)

    st.markdown("#### Results")
    st.write(f"A value above 0.5  means you're healthy, a value below 0.5 \
             means you may be at risk!")
    st.write(f"With the ECG signal above, my model predicted a score of ... {prediction[0][0]:.4f}")

    if binary_array == 1:
        
        st.markdown(
            "#### <div style='text-align: center'> \
                Your heart looks great! 🎉 </div>", 
                unsafe_allow_html=True
            )
        
    else: 
        st.markdown(
            "#### <div style='text-align: center'> \
                We predicted something unusual! Consider booking an \
                appointment with your doctor for a checkup soon! 🚨 </div>", 
                unsafe_allow_html=True \
            )
