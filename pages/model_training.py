"""
Model Training and Comparison Page.
Allows training Linear Regression, Decision Tree, and Random Forest models,
displays comparative metrics, automatic best model selection, and detailed diagnostic charts.
"""

import streamlit as st
import pandas as pd
from src.train_model import (
    train_and_evaluate_all,
    save_model,
    extract_feature_importances
)
from src.visualization import (
    plot_model_comparison,
    plot_actual_vs_predicted,
    plot_feature_importance
)

def render_model_training_page(df):
    """Renders the Model Training and Comparison page."""
    st.title("🤖 Model Training & Comparison")
    st.markdown("""
        Train and evaluate three distinct Machine Learning regression algorithms:
        **Linear Regression**, **Decision Tree Regressor**, and **Random Forest Regressor**.
        Compare their performance metrics on an unseen test dataset and save the optimal model.
    """)

    # Hyperparameter & Train-Test Split Configuration
    with st.sidebar.expander("⚙️ Training Settings", expanded=False):
        test_size = st.slider("Test Set Split Size", min_value=0.15, max_value=0.35, value=0.20, step=0.05,
                              help="Proportion of the dataset reserved for testing model generalization.")
        random_state = st.number_input("Random State Seed", value=42, step=1,
                                       help="Seed for reproducible train/test splitting and forest bagging.")

    # Training Action Button
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        train_clicked = st.button("🚀 Train All Models", type="primary", use_container_width=True)
    with col_info:
        st.write("Click above to split the data, train pipelines, and evaluate test-set metrics.")

    # Check if models have already been trained in session state
    if train_clicked or ('trained_state' not in st.session_state or st.session_state['trained_state'] is None):
        with st.spinner("Training Linear Regression, Decision Tree, and Random Forest pipelines..."):
            trained_state = train_and_evaluate_all(df, test_size=test_size, random_state=random_state)
            st.session_state['trained_state'] = trained_state
            
            # Automatically save best model to disk
            save_model(
                pipeline=trained_state['best_pipeline'],
                model_name=trained_state['best_model_name'],
                metrics_dict=trained_state['comparison_df'].to_dict(orient='records'),
                metadata={
                    "test_size": test_size,
                    "random_state": random_state,
                    "consumption_90th_percentile": trained_state['consumption_90th_percentile'],
                    "consumption_mean": trained_state['consumption_mean']
                }
            )
        st.success(f"🎉 All models trained successfully! Optimal Model Selected: **{trained_state['best_model_name']}**")
    else:
        trained_state = st.session_state['trained_state']

    comparison_df = trained_state['comparison_df']
    best_model_name = trained_state['best_model_name']

    # 1. Best Model Highlight Banner
    best_row = comparison_df[comparison_df['Model'] == best_model_name].iloc[0]
    st.markdown(f"""
        <div style="background-color: #ecfdf5; border-left: 5px solid #10b981; padding: 1rem 1.25rem; border-radius: 8px; margin: 1rem 0;">
            <h4 style="color: #065f46; margin: 0 0 0.25rem 0;">🏆 Selected Best Model: {best_model_name}</h4>
            <p style="color: #047857; margin: 0;">
                Selected based on highest generalization ability on the unseen test set:
                <strong>R² = {best_row['R2_Score']:.4f}</strong> | 
                <strong>RMSE = {best_row['RMSE']:.2f} kWh</strong> | 
                <strong>MAE = {best_row['MAE']:.2f} kWh</strong>
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 2. Model Comparison Table
    st.subheader("📊 Regression Metrics Comparison Table")
    
    # Styled dataframe
    st.dataframe(
        comparison_df.style.highlight_max(subset=['R2_Score'], color='#bbf7d0')
                           .highlight_min(subset=['MAE', 'MSE', 'RMSE'], color='#bbf7d0'),
        use_container_width=True
    )

    # Academic clarification on R2 vs "Accuracy"
    with st.expander("📚 Academic Note: Why R² is NOT simply 'Accuracy' in Regression", expanded=False):
        st.markdown("""
        In academic viva and Machine Learning theory:
        * **Classification** problems output discrete class labels and measure **Accuracy** ($\\% = \\frac{\\text{Correct}}{\\text{Total}}$).
        * **Regression** problems predict continuous numeric values (like electricity consumption in kWh). 
        * Instead of "Accuracy", we use **$R^2$ (Coefficient of Determination)**, which measures the proportion of variance in the target variable explained by the model:
          $$R^2 = 1 - \\frac{\\sum (y_i - \\hat{y}_i)^2}{\\sum (y_i - \\bar{y})^2}$$
        * **MAE (Mean Absolute Error):** Average magnitude of absolute errors in kWh ($|y - \\hat{y}|$). Easy to interpret.
        * **RMSE (Root Mean Squared Error):** Square root of mean squared errors. Heavily penalizes large outlier errors because errors are squared before averaging.
        """)

    st.markdown("---")

    # 3. Chart 5: Model Performance Comparison Chart
    st.subheader("Chart 5: Comparative Performance Visualizer")
    fig_comp = plot_model_comparison(comparison_df)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")

    # 4. Chart 4: Actual vs Predicted Scatter Plot
    st.subheader("Chart 4: Actual vs. Predicted Consumption")
    selected_eval_model = st.selectbox(
        "Select Model to Inspect Predictions:",
        options=list(trained_state['pipelines'].keys()),
        index=list(trained_state['pipelines'].keys()).index(best_model_name)
    )
    
    y_test = trained_state['y_test']
    y_pred = trained_state['test_predictions'][selected_eval_model]
    
    fig_act_pred = plot_actual_vs_predicted(y_test, y_pred, model_name=selected_eval_model)
    st.plotly_chart(fig_act_pred, use_container_width=True)
    st.caption("Points closer to the dashed 45° diagonal line indicate higher prediction accuracy with minimal residuals.")

    st.markdown("---")

    # 5. Chart 6: Feature Importance Analysis
    st.subheader("Chart 6: Feature Importance / Relative Effects")
    selected_pipeline = trained_state['pipelines'][selected_eval_model]
    df_imp = extract_feature_importances(selected_pipeline, selected_eval_model)

    if df_imp is not None:
        fig_imp = plot_feature_importance(df_imp, title=f"Feature Contributions ({selected_eval_model})")
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption("Displays which input features had the highest influence when making predictions.")
    else:
        st.info("Feature importance is not available for this model type.")

    # Persistence Confirmation
    st.markdown("---")
    c_save1, c_save2 = st.columns([1, 4])
    with c_save1:
        if st.button("💾 Re-Save Best Model", use_container_width=True):
            save_model(
                pipeline=trained_state['best_pipeline'],
                model_name=trained_state['best_model_name'],
                metrics_dict=trained_state['comparison_df'].to_dict(orient='records')
            )
            st.success("Model saved to `models/trained_model.pkl`!")
    with c_save2:
        st.caption("The optimal model is serialized with its preprocessing pipeline and ready for the **Predict Consumption** page.")
