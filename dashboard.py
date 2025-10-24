import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import torch
    import joblib
    import torch.nn as nn
    class LSTM_ts(nn.Module):
        def __init__(self, input_size = 1, hidden_size = 64, num_layers = 2, outputs_size = 1, dropout = 0.2):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm  = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )

            self.fc = nn.Linear(hidden_size, outputs_size)
        
        def forward (self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

            out, _ = self.lstm(x, (h0, c0))
            out = out[:, -1, :]
            out = self.fc(out)

            return out
    torch_available = True
except ImportError:
    torch_available = False

# Page configurations
st.set_page_config(
    page_title="Time Serie Vehicle Registration",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Enhanced CSS for better visibility and UX
with open("src/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    filtered_data = pd.read_parquet("Data/dash.parquet")
    return filtered_data

data = load_data()
data["Date"] = pd.to_datetime(data["Date"])
vehicle_types = data['Vehicle Type'].unique().tolist()

min_date = data['Date'].min()
max_date = data['Date'].max()

st.title("🚗 Vehicle Registration Time Series Analysis")

st.markdown("""
    <p>This dashboard provides an interactive analysis of vehicle registration trends over time. Explore the data using the filters below and visualize trends with the provided charts.</p>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("F i l t e r s")

vehicle_type = st.sidebar.multiselect(
    "Select Vehicle Type:",
    options=vehicle_types,
    default=["Total"]
)

st.sidebar.divider()

start_date = st.sidebar.date_input(
    "Start date", 
    value=min_date,
    min_value=min_date,
    max_value=max_date
)
end_date = st.sidebar.date_input(
    "End date", 
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

if start_date > end_date:
    st.sidebar.error("⚠️ Start date cannot be after end date")
    filtered_data = data
else:
    filtered_data = data["Date"].between(start_date, end_date)
    filtered_data = data.loc[filtered_data & data['Vehicle Type'].isin(vehicle_type)]

# Analysis
if not vehicle_type:
    st.warning("⚠️ Please select at least one vehicle type to display the charts.")
else:
    analysis_type = st.selectbox(
        "Select Analysis Type:",
        options=["Time Series", "Decomposition", "ACF/PACF", "Forecasting"],
        key="analysis_type"
    )

    if analysis_type == "Time Series":
        fig = px.line(
            filtered_data,
            x="Date",
            y="Registers",
            color="Vehicle Type",
            title="📈 Vehicle Registrations Over Time"
        )
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=-0.25,
            text="⚠️ Note: Some vehicle types have very low registrations and may appear as zero in the chart.",
            showarrow=False,
            font=dict(size=12, color="white"),
            opacity=0.8
        )

        fig.update_layout(legend_title_text='Vehicle Type')
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Decomposition":

        vehicle_types = filtered_data['Vehicle Type'].unique()
        n_types = len(vehicle_types)

        subplot_titles = []

        for vehicle in vehicle_types:
            subplot_titles.extend([
                f"{vehicle} - Original",
                f"{vehicle} - Trend",
                f"{vehicle} - Seasonality",
                f"{vehicle} - Residuals"
            ])

        fig = make_subplots(
            rows=n_types, cols=4,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.05,
            vertical_spacing=0.15
        )

        for i, vehicle in enumerate(vehicle_types, start=1):
            serie = (
                filtered_data[filtered_data['Vehicle Type'] == vehicle]
                .set_index('Date')['Registers']
            )

            results = seasonal_decompose(serie, model='additive', period=12)

            # Add Original
            fig.add_trace(
                go.Scatter(x=serie.index, y=serie, name=f'{vehicle} Original'), 
                row=i, col=1
            )

            # Add Trend
            fig.add_trace(
                go.Scatter(x=serie.index, y=results.trend, name=f'{vehicle} Trend'),
                row=i, col=2
            )

            # Add Seasonality
            fig.add_trace(
                go.Scatter(x=serie.index, y=results.seasonal, name=f'{vehicle} Seasonality'),
                row=i, col=3
            )

            # Add Residuals
            fig.add_trace(
                go.Scatter(x=serie.index, y=results.resid, name=f'{vehicle} Residuals'),
                row=i, col=4
            )

        fig.update_layout(
            height=300 * n_types, width=1200,
            title="🔎 Seasonal Decomposition by Vehicle Type",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "ACF/PACF":

        vehicle_types = filtered_data["Vehicle Type"].unique()
        n_row = len(vehicle_types)

        subplot_titles = []

        for vehicle in vehicle_types:
            subplot_titles.extend([
                f'{vehicle} - ACF',
                f'{vehicle} - PACF'
            ])

        fig = make_subplots(
            rows=n_row, cols=2,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.05,
            vertical_spacing=0.15
        )

        for i, vehicle in enumerate(vehicle_types, start=1):
            serie = (
                filtered_data[filtered_data['Vehicle Type'] == vehicle]
                .set_index('Date')['Registers']
            )

            acf_vals = acf(serie, nlags=36)
            pacf_vals = pacf(serie, nlags=36)

            # Add ACF
            fig.add_trace(
                go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name=f'{vehicle} ACF'),
                row=i, col=1
            )

            # Add PACF
            fig.add_trace(
                go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, name=f'{vehicle} PACF'),
                row=i, col=2
            )

        fig.update_layout(
            height=300*n_row, width=1000,
            title_text="🔎 ACF & PACF by Vehicle Type",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Forecasting":
            df_test = {}
            for vehicle in filtered_data['Vehicle Type'].unique():
                path = f'Outputs/Predictions/eval_pred_{vehicle.replace(" ", "_").lower()}.parquet'
                df_test[vehicle] = pd.read_parquet(path)

                df_vehicle = df_test[vehicle]
                rmse = np.sqrt(np.mean((df_vehicle['Pred'] - df_vehicle['Real'])**2))

                df_vehicle['Upper'] = df_vehicle['Pred'] + 1.96 * rmse
                df_vehicle['Lower'] = df_vehicle['Pred'] - 1.96 * rmse

                st.markdown(f"### 📊 Forecasting Results for {vehicle}")

                fig = px.line(
                    df_vehicle,
                    x='Date',
                    y=['Real', 'Pred'],
                    labels={'value': 'Registrations', 'variable': 'Legend'},
                    title=f'Forecast vs Actual for {vehicle}'
                )

                fig.add_traces([
                    go.Scatter(
                        x=df_vehicle['Date'], y=df_vehicle['Upper'],
                        mode='lines',
                        line=dict(width=0),
                        name='Upper Bound',
                        showlegend=False
                    ),
                    go.Scatter(
                        x=df_vehicle['Date'], y=df_vehicle['Lower'],
                        mode='lines',
                        fill='tonexty',
                        line=dict(width=0),
                        fillcolor='rgba(0,100,80,0.2)',
                        name='Confidence Interval',
                        showlegend=True
                    )
                ])

                fig.update_layout(legend_title_text='Legend')
                st.plotly_chart(fig, use_container_width=True)

            

