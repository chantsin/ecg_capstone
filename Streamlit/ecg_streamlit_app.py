# ecg_streamlit_app

### IMPORT LIBRARIES
import numpy as np
# import pandas as pd 
# import matplotlib.pyplot as plt
import streamlit as st 
# import joblib
# import wfdb
# import altair as alt
# import plotly.graph_objects as go 

# from utils import ecg_cleaning as c

# Set app title 

main_title = 'ECG Classification'

st.set_page_config(
    page_title=main_title, 
    page_icon="🫀",
    layout='centered')


st.sidebar.markdown("# About My Capstone")
st.sidebar.markdown("My capstone project utilizes machine learning (ML) \
                    algorithms to extract key features that distinguish \
                    between a normal and abnormal ECG. ")
st.sidebar.markdown("Using the model, I am able to predict whether new ECGs \
                    have any abnormalities shown in them and tell you how your \
                    heart is doing. ")
st.sidebar.markdown("Utilizing Recurrent Neural Networks or RNNs as my model, \
                    I am able to predict ECGs with an accuracy of over 80%, \
                    giving confidence to users about their health.")

st.title('ECG Classification via Fourier Transforms 🫀')

st.markdown("""
            
In this project I will walk you through how I used data science \
tools to solve a problem in the medical field. Let's dive right \
into it!

### Introduction

Biomedical signals have always played an important role in the medical field. \
One type of biomedical signal that you may be familiar with is an ECG or \
electrocardiogram. An ECG looks at how our heart functions by recording our \
heart rate and heart rhythm. Using these records, physicians can come to \
conclusions about whether a person is healthy or not, and if its the latter \
they can come up with a proper treatment outcome for the patient. 
            
### Problem Space

Traditionally, ECG signals are interpreted solely based on human \
readibility, and using these results doctors can pinpoint the next steps \
needed. However, this poses a risk in that although doctors are very \
professional in their line of work, they can make mistakes too. This can lead \
to misdiagnosis resulting in patients not receiving proper treatment. To \
counteract this issue, I proposed to use data science tools, specifically \
machine learning (ML) algorithms to predict ECG diagnostics. 
""")


# Create a sample ECG signal 
path = 'data/physionet.org/files/ptb-xl/1.0.3/records100/00000/00004_lr'
#ecg = ''
#file = path + ecg
metadata = pd.read_csv('Streamlit/cleaned_metadata.csv', index_col=0)

signals, fields = wfdb.rdsamp(path) #, channels=[1])

sampling_frequency = fields['fs']  # Sampling frequency
sig_len = fields['sig_len'] # Signal length 
sig_name = fields['sig_name'] # Lead

time = np.arange(0, sig_len) / sampling_frequency

# First we get the sampling interval 
dt = 1/sampling_frequency

# Flatten the signal from 2D to 1D 
signals = signals.flatten()

# FFT returns a result not centered at zero, therefore we need to fftshift it to zero
signal_fft = np.fft.fftshift(np.fft.fft(signals) * dt)
freq = np.fft.fftshift(np.fft.fftfreq(sig_len, dt))

# Baseline removal
signal_bl = c.baseline_removal(signals, freq_start=0.1, freq_stop=1.5)

# Plotting sample ECG signal and the denoised version 
samp_ecg = pd.DataFrame(
    {
        'Time (seconds)': time,
        'Original ECG': signals, 
        'Denoised ECG': signal_bl,
        'Amplitude': np.abs(signal_fft)
    }
)

st.markdown("#### Example ECG Signal")

# chart1 = alt.Chart(samp_ecg).mark_line().encode(
#     x=alt.X('Time (seconds)'),
#     y=alt.Y('Original ECG').title('Amplitude (mV)'),
#     color=alt.value('red')
# ).interactive()

# chart2 = alt.Chart(samp_ecg).mark_line().encode(
#     x=alt.X('Time (seconds)'),
#     y=alt.Y('Denoised ECG').title('Amplitude (mV)'),
#     color=alt.value('blue'),
# ).interactive()

# chart = alt.layer(chart1, chart2)

# st.altair_chart(chart, theme="streamlit", use_container_width=True)

# Create traces for the line graphs
trace1 = go.Scatter(x=time, 
                    y=signals, 
                    mode='lines', 
                    name='Original ECG',
                    line={'color': 'blue'})
trace2 = go.Scatter(x=time, 
                    y=signal_bl, 
                    mode='lines', 
                    name='Denoised ECG',
                    line={'color': 'red'})

layout = go.Layout(
    xaxis=dict(
        range=[0, 10],
        dtick=1,
        title='Time (seconds)',
        showgrid=True,  # Show gridlines on the x-axis
        gridcolor='rgba(0,0,0,0.1)',  # Set gridline color (optional)
        gridwidth=1  # Set gridline width (optional)
    ),
    yaxis=dict(
        range=[-1.5, 2],
        dtick=1,
        title='Amplitude (mV)',
        showgrid=True,  # Show gridlines on the y-axis
        gridcolor='rgba(0,0,0,0.1)',  # Set gridline color (optional)
        gridwidth=1  # Set gridline width (optional)
    )
)

# Create a Plotly figure with both line graphs
fig = go.Figure([trace1, trace2], layout=layout)

# Display the figure using Streamlit
st.plotly_chart(fig, theme="streamlit", use_container_width=True)

######################### INTRO TO FOURIER TRANSFORMS ##########################

st.markdown("""### A Brief Introduction to Fourier Transforms""")

st.markdown("""

Let me give you a brief introduction into what Fourier Transforms is and why 
we are talking about it! \

Fourier Transforms is a mathematical tool that is used in signal processing, 
and in our case the signal is the ECG itself. 
            
Often times when we are taking medical readings these signals can be obscured by
noise, that is signals not coming from the source we are trying to measure. When
recording ECG signals this might be in the form of the persons arm moving or 
body shifting when they are being recorded, or it can also come from our lungs
when we breath. These types of noise can make interpreting an ECG hard, and this
is where Fourier Transforms come in. 
            
Knowing that these signals have a certain pattern to them, we can explicitly 
remove them from our signal through the procedure of a Fourier Analysis. Signals
are usually taken over time. What Fourier Analysis does is look at what 
frequencies make up that signal instead of looking at it over time. Since we 
know that certain frequencies are associated with those noise, we can remove 
those frequencies being looking at the signal again over time.
""")

st.markdown("#### Fourier Transform of ECG Signal")

trace3 = go.Scatter(x=freq, 
                    y=np.abs(signal_fft), 
                    mode='lines',
                    line={'color': 'blue'})

layout2 = go.Layout(
    xaxis=dict(
        title='Frequency (Hz)',
        showgrid=True,  # Show gridlines on the x-axis
        gridcolor='rgba(0,0,0,0.1)',  # Set gridline color (optional)
        gridwidth=1  # Set gridline width (optional)
    ),
    yaxis=dict(
        title='Amplitude',
        showgrid=True,  # Show gridlines on the y-axis
        gridcolor='rgba(0,0,0,0.1)',  # Set gridline color (optional)
        gridwidth=1  # Set gridline width (optional)
    )
)

# Create a Plotly figure with both line graphs
fig2 = go.Figure([trace3], layout=layout2)

# Display the figure using Streamlit
st.plotly_chart(fig2, theme="streamlit", use_container_width=True)


