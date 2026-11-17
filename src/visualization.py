"""
Visualization Module for Electricity Consumption Predictor.
Generates clean, interactive Plotly charts for exploratory data analysis,
model evaluation, residual analysis, and feature importances.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Harmonious, modern color palette
PRIMARY_COLOR = "#2563EB"   # Royal Blue
ACCENT_COLOR = "#10B981"    # Emerald Green
WARN_COLOR = "#F59E0B"      # Amber
DANGER_COLOR = "#EF4444"    # Rose Red
BG_TEMPLATE = "plotly_white"

def plot_consumption_over_time(df):
    """Chart 1: Electricity consumption over time."""
    df_plot = df.copy()
    if 'Date' in df_plot.columns:
        df_plot['Date'] = pd.to_datetime(df_plot['Date'])
        df_plot = df_plot.sort_values('Date')
        x_col = 'Date'
    else:
        x_col = df_plot.index
        
    fig = px.line(
        df_plot,
        x=x_col,
        y='Electricity_Consumption',
        title="⚡ Electricity Consumption Over Time",
        labels={'Electricity_Consumption': 'Consumption (kWh)', x_col: 'Timeline'},
        template=BG_TEMPLATE,
        color_discrete_sequence=[PRIMARY_COLOR]
    )
    
    # Add a 30-day moving average trendline
    if len(df_plot) > 30 and 'Date' in df_plot.columns:
        df_plot['Rolling_Avg'] = df_plot['Electricity_Consumption'].rolling(window=30, center=True).mean()
        fig.add_trace(go.Scatter(
            x=df_plot['Date'],
            y=df_plot['Rolling_Avg'],
            mode='lines',
            name='30-Day Moving Average',
            line=dict(color=WARN_COLOR, width=2.5, dash='dash')
        ))
        
    fig.update_layout(
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_temp_vs_consumption(df):
    """Chart 2: Temperature vs Electricity Consumption with Humidity color scale."""
    fig = px.scatter(
        df,
        x='Temperature',
        y='Electricity_Consumption',
        color='Humidity',
        size='Appliance_Usage',
        hover_data=['Day_of_Week', 'Number_of_People'],
        title="🌡️ Temperature vs. Electricity Consumption (Bubble size = Appliance Usage)",
        labels={
            'Temperature': 'Temperature (°C)',
            'Electricity_Consumption': 'Consumption (kWh)',
            'Humidity': 'Humidity (%)'
        },
        color_continuous_scale="Viridis",
        template=BG_TEMPLATE
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def plot_consumption_by_day(df):
    """Chart 3: Average consumption by day of week in proper calendar order."""
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_plot = df.copy()
    
    if 'Day_of_Week' not in df_plot.columns and 'Date' in df_plot.columns:
        df_plot['Day_of_Week'] = pd.to_datetime(df_plot['Date']).dt.day_name()
        
    avg_day = df_plot.groupby('Day_of_Week', as_index=False)['Electricity_Consumption'].agg(['mean', 'std']).reset_index()
    # Categorical ordering
    avg_day['Day_of_Week'] = pd.Categorical(avg_day['Day_of_Week'], categories=order, ordered=True)
    avg_day = avg_day.sort_values('Day_of_Week')
    
    fig = px.bar(
        avg_day,
        x='Day_of_Week',
        y='mean',
        error_y='std',
        title="📅 Average Electricity Consumption by Day of Week",
        labels={'Day_of_Week': 'Day of the Week', 'mean': 'Average Consumption (kWh)'},
        color='mean',
        color_continuous_scale="Blues",
        template=BG_TEMPLATE,
        text_auto='.1f'
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def plot_actual_vs_predicted(y_actual, y_predicted, model_name="Best Model"):
    """Chart 4: Actual vs Predicted consumption scatter plot with y=x reference line."""
    df_eval = pd.DataFrame({
        'Actual': y_actual,
        'Predicted': y_predicted,
        'Residual': y_actual - y_predicted
    })
    
    min_val = min(y_actual.min(), y_predicted.min()) * 0.95
    max_val = max(y_actual.max(), y_predicted.max()) * 1.05
    
    fig = px.scatter(
        df_eval,
        x='Actual',
        y='Predicted',
        color='Residual',
        color_continuous_scale='RdYlBu_r',
        title=f"🎯 Actual vs. Predicted Consumption ({model_name})",
        labels={'Actual': 'Actual Consumption (kWh)', 'Predicted': 'Predicted Consumption (kWh)'},
        template=BG_TEMPLATE,
        hover_data=['Residual']
    )
    
    # 45-degree reference line
    fig.add_shape(
        type="line", line=dict(dash='dash', color='gray', width=2),
        x0=min_val, y0=min_val, x1=max_val, y1=max_val
    )
    
    fig.update_layout(
        xaxis=dict(range=[min_val, max_val]),
        yaxis=dict(range=[min_val, max_val]),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def plot_model_comparison(comparison_df):
    """Chart 5: Model performance comparison (MAE, RMSE, R2 Score)."""
    fig = go.Figure()
    
    models = comparison_df['Model'].tolist()
    
    fig.add_trace(go.Bar(
        x=models,
        y=comparison_df['MAE'],
        name='MAE (Lower is Better)',
        marker_color='#3B82F6',
        text=comparison_df['MAE'],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        x=models,
        y=comparison_df['RMSE'],
        name='RMSE (Lower is Better)',
        marker_color='#F59E0B',
        text=comparison_df['RMSE'],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        x=models,
        y=comparison_df['R2_Score'],
        name='R² Score (Higher is Better)',
        marker_color='#10B981',
        text=comparison_df['R2_Score'],
        textposition='auto'
    ))
    
    fig.update_layout(
        barmode='group',
        title="📊 Model Performance Metrics Comparison",
        xaxis_title="Machine Learning Models",
        yaxis_title="Metric Value",
        template=BG_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

def plot_feature_importance(df_imp, title="Feature Importance / Effect"):
    """Chart 6: Feature importance or regression coefficients."""
    # Determine column: 'Importance' (trees) or 'Absolute_Effect' (linear)
    col = 'Importance' if 'Importance' in df_imp.columns else 'Absolute_Effect'
    
    df_sorted = df_imp.sort_values(by=col, ascending=True)
    
    fig = px.bar(
        df_sorted,
        x=col,
        y='Feature',
        orientation='h',
        title=f"🌳 {title}",
        labels={col: 'Relative Importance Score', 'Feature': 'Input Feature'},
        template=BG_TEMPLATE,
        color=col,
        color_continuous_scale='Tealgrn',
        text_auto='.3f'
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def plot_correlation_heatmap(df):
    """Bonus EDA: Correlation heatmap for numerical features."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr().round(2)
    
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='Blues',
        title="🔥 Feature Correlation Matrix",
        template=BG_TEMPLATE
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig
