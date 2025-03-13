import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
np.random.seed(42)

# -------------------------------
# Function: Retrieve Stock Free Cash Flow from yfinance
# -------------------------------
def get_base_fcf(ticker):
    stock = yf.Ticker(ticker)
    try:
        info = stock.info
    except Exception as e:
        print(f"Error fetching info for {ticker}: {e}")
        return None
    
    # Retrieve free cash flow from the info dictionary
    base_fcf = info.get("freeCashflow", None)
    if base_fcf is None or base_fcf <= 0:
        print(f"No valid free cash flow data available for {ticker}.")
        return None
    
    print(f"Retrieved base free cash flow for {ticker}: ${base_fcf:,.0f}")
    return base_fcf

# -------------------------------
# Monte Carlo DCF Simulation Function
# -------------------------------
def monte_carlo_dcf(base_fcf, forecast_years=5, n_iterations=5000,
                    growth_rate_mean=0.05, growth_rate_std=0.02,
                    discount_rate_mean=0.08, discount_rate_std=0.01,
                    terminal_growth_mean=0.02, terminal_growth_std=0.005):
    dcf_values = []  # Store simulated enterprise values
    
    for _ in range(n_iterations):
        # Sample uncertain parameters from normal distributions
        growth_rate = np.random.normal(growth_rate_mean, growth_rate_std)
        discount_rate = np.random.normal(discount_rate_mean, discount_rate_std)
        terminal_growth = np.random.normal(terminal_growth_mean, terminal_growth_std)
        
        # Ensure discount rate is greater than terminal growth to avoid division issues
        if discount_rate <= terminal_growth:
            discount_rate = terminal_growth + 0.01
        
        # Project and discount free cash flows over the forecast period
        discounted_fcfs = []
        for year in range(1, forecast_years + 1):
            projected_fcf = base_fcf * (1 + growth_rate) ** year
            discounted_fcf = projected_fcf / ((1 + discount_rate) ** year)
            discounted_fcfs.append(discounted_fcf)
        
        # Terminal value calculation using the Gordon Growth Model
        last_projected_fcf = base_fcf * (1 + growth_rate) ** forecast_years
        terminal_value = last_projected_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
        discounted_terminal_value = terminal_value / ((1 + discount_rate) ** forecast_years)
        
        # Sum the discounted free cash flows and the discounted terminal value
        enterprise_value = sum(discounted_fcfs) + discounted_terminal_value
        dcf_values.append(enterprise_value)
    
    return np.array(dcf_values)

# -------------------------------
# Visualization and Summary Functions
# -------------------------------
def summarize_and_plot(dcf_values):
    mean_val = np.mean(dcf_values)
    median_val = np.median(dcf_values)
    std_val = np.std(dcf_values)
    perc_5 = np.percentile(dcf_values, 5)
    perc_95 = np.percentile(dcf_values, 95)
    
    print("Monte Carlo DCF Valuation Simulation Results")
    print(f"Iterations: {len(dcf_values)}")
    print(f"Mean Enterprise Value: ${mean_val:,.2f}")
    print(f"Median Enterprise Value: ${median_val:,.2f}")
    print(f"Standard Deviation: ${std_val:,.2f}")
    print(f"5th Percentile: ${perc_5:,.2f}")
    print(f"95th Percentile: ${perc_95:,.2f}")
    
    # Plot histogram of enterprise values
    plt.figure(figsize=(10, 6))
    plt.hist(dcf_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title("Monte Carlo Simulation of DCF Valuation")
    plt.xlabel("Enterprise Value (USD)")
    plt.ylabel("Frequency")
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: ${mean_val:,.0f}')
    plt.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: ${median_val:,.0f}')
    plt.legend()
    plt.grid(True)
    plt.show()

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    # Specify the ticker for analysis
    ticker = "AAPL"  # Change to desired ticker
    
    # Retrieve base free cash flow from the stock's data
    base_fcf = get_base_fcf(ticker)
    if base_fcf is None:
        print("Exiting due to lack of FCF data.")
    else:
        # Run the Monte Carlo DCF simulation
        dcf_values = monte_carlo_dcf(base_fcf,
                                     forecast_years=5,
                                     n_iterations=5000,
                                     growth_rate_mean=0.05,
                                     growth_rate_std=0.02,
                                     discount_rate_mean=0.08,
                                     discount_rate_std=0.01,
                                     terminal_growth_mean=0.02,
                                     terminal_growth_std=0.005)
        # Summarize and visualize the results
        summarize_and_plot(dcf_values)
