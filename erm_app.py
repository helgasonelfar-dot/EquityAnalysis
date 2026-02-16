import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- ERM Valuation Function ---
def calculate_erm_valuation_adjustable(latest_bvps, normalized_roe, cost_of_equity, bvps_growth_rate, forecast_period, stable_growth_rate):
    '''
    Calculates the intrinsic value per share using the Excess Return Model (ERM).

    Args:
        latest_bvps (float): The most recent Book Value Per Share.
        normalized_roe (float): The normalized Return on Equity.
        cost_of_equity (float): The Cost of Equity (Ke).
        bvps_growth_rate (float): The historical BVPS growth rate for explicit forecast BVPS projection.
        forecast_period (int): The number of years for the explicit forecast period.
        stable_growth_rate (float): The stable growth rate for the terminal value calculation.

    Returns:
        float: The calculated intrinsic value per share.
    '''

    # Check for Gordon Growth Model denominator validity
    if cost_of_equity <= stable_growth_rate:
        return np.nan

    # Forecast excess returns per share for the explicit forecast period
    forecasted_bvps = [latest_bvps] # BVPS at the start of each year (BVPS_0 is latest_bvps)
    pv_explicit_excess_returns = 0.0

    for year in range(1, forecast_period + 1):
        excess_return_this_year = (normalized_roe - cost_of_equity) * forecasted_bvps[-1]
        discount_factor = 1 / ((1 + cost_of_equity) ** year)
        pv_explicit_excess_returns += excess_return_this_year * discount_factor
        projected_next_bvps = forecasted_bvps[-1] * (1 + bvps_growth_rate)
        forecasted_bvps.append(projected_next_bvps)

    bvps_at_forecast_end = forecasted_bvps[-1]
    excess_return_year_after_forecast = (normalized_roe - cost_of_equity) * bvps_at_forecast_end
    terminal_excess_value = excess_return_year_after_forecast / (cost_of_equity - stable_growth_rate)
    pv_terminal_excess_value = terminal_excess_value / ((1 + cost_of_equity) ** forecast_period)
    intrinsic_value_per_share = latest_bvps + pv_explicit_excess_returns + pv_terminal_excess_value

    return intrinsic_value_per_share


# --- Data for Streamlit App (Simulated/Hardcoded based on previous steps) ---
# In a real application, this data would be dynamically loaded.
bank_data = {
    'ISB.IC': {
        'latest_bvps': 129.3982503276895,
        'normalized_roe': 0.11116378887759569,
        'bvps_growth_rate': 0.0373,
        'current_share_price': 139.0
    },
    'KVIKA.IC': {
        'latest_bvps': 20.24873169222748,
        'normalized_roe': 0.08432150586386007,
        'bvps_growth_rate': 0.0448,
        'current_share_price': 17.9
    },
    'ARION.IC': {
        'latest_bvps': 150.31228092700118,
        'normalized_roe': 0.13549258456015068,
        'bvps_growth_rate': 0.0213,
        'current_share_price': 197.0
    }
}

# Initial placeholder beta values. In a real app, these would be user inputs or calculated.
initial_beta_values = {
    'ISB.IC': 1.1,
    'KVIKA.IC': 1.2,
    'ARION.IC': 1.0
}


def conduct_sensitivity_analysis(bank_ticker, initial_params, roe_range, ke_range, sgr_range):
    '''
    Conducts sensitivity analysis for a given bank across specified ranges of ROE, Ke, and SGR.
    '''
    sensitivity_results = []
    for roe_val in roe_range:
        for ke_val in ke_range:
            for sgr_val in sgr_range:
                intrinsic_val = calculate_erm_valuation_adjustable(
                    latest_bvps=initial_params['latest_bvps'],
                    normalized_roe=roe_val,
                    cost_of_equity=ke_val,
                    bvps_growth_rate=initial_params['bvps_growth_rate'],
                    forecast_period=initial_params['forecast_period'],
                    stable_growth_rate=sgr_val
                )
                sensitivity_results.append({
                    'Ticker': bank_ticker,
                    'Test_ROE': roe_val,
                    'Test_Ke': ke_val,
                    'Test_SGR': sgr_val,
                    'Intrinsic_Value': intrinsic_val
                })
    return pd.DataFrame(sensitivity_results)

def main():
    st.set_page_config(layout="wide")
    st.title('Excess Return Model (ERM) Valuation for Icelandic Banks')

    st.markdown('''
    This application allows you to perform Excess Return Model (ERM) valuation for selected Icelandic banks.
    Adjust the key valuation parameters below to see their impact on the intrinsic value.
    ''')

    st.subheader("Valuation Inputs")

    # Global Inputs (Sidebar)
    st.sidebar.header("Global Valuation Inputs")
    risk_free_rate = st.sidebar.slider('Risk-Free Rate (Rf)', min_value=0.00, max_value=0.10, value=0.035, step=0.001, format='%.3f')
    equity_risk_premium = st.sidebar.slider('Equity Risk Premium (ERP)', min_value=0.01, max_value=0.10, value=0.055, step=0.001, format='%.3f')
    forecast_period = st.sidebar.slider('Explicit Forecast Period (Years)', min_value=1, max_value=10, value=5, step=1)
    stable_growth_rate = st.sidebar.slider('Stable Growth Rate (g)', min_value=0.00, max_value=0.05, value=0.025, step=0.001, format='%.3f')

    # Bank-Specific Inputs
    st.header("Bank-Specific Inputs")
    cols = st.columns(len(bank_data))

    user_beta_values = {}
    user_normalized_roes = {}

    calculated_cost_of_equities = {}

    for i, (ticker, data) in enumerate(bank_data.items()):
        with cols[i]:
            st.subheader(f"{ticker}")
            st.write(f"Current Share Price: {data['current_share_price']:.2f}")
            st.write(f"Latest BVPS: {data['latest_bvps']:.2f}")
            st.write(f"Historical BVPS Growth: {data['bvps_growth_rate']:.2%}")

            user_beta_values[ticker] = st.number_input(
                f'Beta for {ticker}',
                min_value=0.5,
                max_value=2.0,
                value=initial_beta_values.get(ticker, 1.0),
                step=0.05,
                format='%.2f',
                key=f'beta_{ticker}'
            )

            user_normalized_roes[ticker] = st.number_input(
                f'Normalized ROE for {ticker}',
                min_value=0.00,
                max_value=0.25,
                value=data['normalized_roe'],
                step=0.005,
                format='%.3f',
                key=f'roe_{ticker}'
            )

            cost_of_equity = risk_free_rate + (user_beta_values[ticker] * equity_risk_premium)
            calculated_cost_of_equities[ticker] = cost_of_equity
            st.write(f"Calculated Ke: {cost_of_equity:.3%}")

    st.markdown("--- ") # Separator

    st.subheader("Valuation Results")
    valuation_summary_data = []
    all_sensitivity_dfs = []

    for ticker, data in bank_data.items():
        current_ke = calculated_cost_of_equities[ticker]
        current_roe = user_normalized_roes[ticker]

        # This is a local copy of bank_data for use within the loop, including current user inputs
        data_for_current_valuation = data.copy() # Avoid modifying the original bank_data directly
        data_for_current_valuation['normalized_roe'] = current_roe
        data_for_current_valuation['cost_of_equity'] = current_ke
        data_for_current_valuation['forecast_period'] = forecast_period # Pass current forecast period
        data_for_current_valuation['stable_growth_rate'] = stable_growth_rate # Pass current stable growth rate

        intrinsic_value = calculate_erm_valuation_adjustable(
            latest_bvps=data_for_current_valuation['latest_bvps'],
            normalized_roe=current_roe,
            cost_of_equity=current_ke,
            bvps_growth_rate=data_for_current_valuation['bvps_growth_rate'],
            forecast_period=forecast_period,
            stable_growth_rate=stable_growth_rate
        )

        if np.isnan(intrinsic_value):
            intrinsic_value_str = "N/A (Ke <= g)"
            difference = "N/A"
        else:
            intrinsic_value_str = f"{intrinsic_value:.2f}"
            difference = f"{intrinsic_value - data['current_share_price']:.2f}"

        valuation_summary_data.append({
            'Bank': ticker,
            'Current Price': f"{data['current_share_price']:.2f}",
            'Intrinsic Value': intrinsic_value_str,
            'Difference (Intrinsic - Current)': difference,
            'Normalized ROE': f"{current_roe:.2%}",
            'Cost of Equity (Ke)': f"{current_ke:.2%}",
            'Stable Growth Rate': f"{stable_growth_rate:.2%}"
        })

        # Define sensitivity ranges based on current user inputs for this bank
        roe_sens_range = np.linspace(data_for_current_valuation['normalized_roe'] - 0.02, data_for_current_valuation['normalized_roe'] + 0.02, 5)
        ke_sens_range = np.linspace(data_for_current_valuation['cost_of_equity'] - 0.01, data_for_current_valuation['cost_of_equity'] + 0.01, 5)
        sgr_sens_range = np.linspace(data_for_current_valuation['stable_growth_rate'] - 0.01, data_for_current_valuation['stable_growth_rate'] + 0.01, 5)

        # Ensure ranges are not negative
        roe_sens_range[roe_sens_range < 0] = 0
        ke_sens_range[ke_sens_range < 0] = 0
        sgr_sens_range[sgr_sens_range < 0] = 0

        # Conduct sensitivity analysis for each bank using current inputs
        sensitivity_df_bank = conduct_sensitivity_analysis(
            ticker,
            data_for_current_valuation, # Pass the locally updated data
            roe_sens_range,
            ke_sens_range,
            sgr_sens_range
        )
        all_sensitivity_dfs.append(sensitivity_df_bank)

    valuation_summary_df = pd.DataFrame(valuation_summary_data)
    st.table(valuation_summary_df.set_index('Bank'))

    # --- Interactive Sensitivity Analysis ---
    st.header("Interactive Sensitivity Analysis")
    selected_banks = st.multiselect(
        'Select banks for sensitivity analysis',
        options=list(bank_data.keys()),
        default=list(bank_data.keys())
    )

    # Combine all sensitivity dataframes
    if all_sensitivity_dfs:
        full_sensitivity_df = pd.concat(all_sensitivity_dfs, ignore_index=True)
        # Drop rows where intrinsic_value is NaN due to Ke <= SGR
        full_sensitivity_df = full_sensitivity_df.dropna(subset=['Intrinsic_Value'])

        if selected_banks:
            filtered_sensitivity_df = full_sensitivity_df[full_sensitivity_df['Ticker'].isin(selected_banks)]

            for ticker in selected_banks:
                st.markdown(f"#### Sensitivity for {ticker}")
                bank_df = filtered_sensitivity_df[filtered_sensitivity_df['Ticker'] == ticker]

                # Define initial_params here for use in plotting sections
                initial_params_for_plotting = {
                    'latest_bvps': bank_data[ticker]['latest_bvps'],
                    'normalized_roe': user_normalized_roes[ticker],
                    'cost_of_equity': calculated_cost_of_equities[ticker],
                    'bvps_growth_rate': bank_data[ticker]['bvps_growth_rate'],
                    'forecast_period': forecast_period,
                    'stable_growth_rate': stable_growth_rate
                }

                # Define numerical ranges for plotting axes based on initial_params_for_plotting
                plot_ke_sens_range_numerical = np.linspace(initial_params_for_plotting['cost_of_equity'] - 0.01, initial_params_for_plotting['cost_of_equity'] + 0.01, 5)
                plot_roe_sens_range_numerical = np.linspace(initial_params_for_plotting['normalized_roe'] - 0.02, initial_params_for_plotting['normalized_roe'] + 0.02, 5)

                # Ensure ranges are not negative
                plot_ke_sens_range_numerical[plot_ke_sens_range_numerical < 0] = 0
                plot_roe_sens_range_numerical[plot_roe_sens_range_numerical < 0] = 0


                # Get the current stable growth rate for this bank, as used in the sensitivity ranges
                current_sgr_for_table = stable_growth_rate # Use the global stable growth rate from the slider

                # Sensitivity of Intrinsic Value to ROE and Ke (fixing SGR at current global value)
                # Filter by the exact current_sgr_for_table, or the closest value in the sgr_sens_range if exact match not present
                sgr_fixed_df = bank_df[np.isclose(bank_df['Test_SGR'], current_sgr_for_table)]

                if not sgr_fixed_df.empty:
                    sensitivity_table = sgr_fixed_df.pivot_table(
                        index='Test_ROE',
                        columns='Test_Ke',
                        values='Intrinsic_Value'
                    )
                    # Convert to percentage strings for display
                    sensitivity_table.index = (sensitivity_table.index * 100).map('{:.2f}% ROE'.format)
                    sensitivity_table.columns = (sensitivity_table.columns * 100).map('{:.2f}% Ke'.format)

                    st.write(f"Intrinsic Value Sensitivity Table (Stable Growth Rate: {current_sgr_for_table:.2%})")
                    st.dataframe(sensitivity_table.map(lambda x: f'{x:.2f}' if not pd.isna(x) else 'N/A'))

                    fig_3d = go.Figure(data=[go.Surface(
                        z=sensitivity_table.values,
                        x=plot_ke_sens_range_numerical,
                        y=plot_roe_sens_range_numerical
                    )])
                    fig_3d.update_layout(
                        title=f'{ticker}: Intrinsic Value Sensitivity (ROE vs. Ke)',
                        scene=dict(
                            xaxis_title='Cost of Equity',
                            yaxis_title='Normalized ROE',
                            zaxis_title='Intrinsic Value'
                        )
                    )
                    st.plotly_chart(fig_3d)
                else:
                    st.info(f"No valid data to display ROE vs Ke sensitivity for {ticker} at the current stable growth rate. Adjust global SGR or ranges.")

                # Impact of Stable Growth Rate (fixing ROE and Ke at current user-defined values)
                sgr_impact_df = bank_df[
                    (np.isclose(bank_df['Test_ROE'], initial_params_for_plotting['normalized_roe'])) &
                    (np.isclose(bank_df['Test_Ke'], initial_params_for_plotting['cost_of_equity']))
                ].sort_values(by='Test_SGR')

                if not sgr_impact_df.empty:
                    fig_line = go.Figure(data=go.Scatter(
                        x=sgr_impact_df['Test_SGR'],
                        y=sgr_impact_df['Intrinsic_Value'],
                        mode='lines+markers',
                        name='Intrinsic Value'
                    ))
                    fig_line.update_layout(
                        title=f'{ticker}: Intrinsic Value vs. Stable Growth Rate (ROE: {initial_params_for_plotting['normalized_roe']:.2%}, Ke: {initial_params_for_plotting['cost_of_equity']:.2%})',
                        xaxis_title='Stable Growth Rate',
                        yaxis_title='Intrinsic Value',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_line)
                else:
                    st.info(f"No valid data to display Stable Growth Rate impact for {ticker}. Adjust ROE/Ke or SGR ranges.")

        else:
            st.info("Select at least one bank to view sensitivity analysis.")
    else:
        st.info("No sensitivity analysis data available.")


if __name__ == '__main__':
    main()

