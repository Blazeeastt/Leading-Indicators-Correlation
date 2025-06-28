import pandas as pd
import glob
import numpy as np

file_paths = glob.glob('*.csv')

close_price_data = {}

for file_path in file_paths:
    try:
        df = pd.read_csv(file_path)

        required_columns = ['date', 'close']
        if not all(col in df.columns for col in required_columns):
            print(f"Skipping {file_path}: missing required columns")
            continue

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)

        if df.empty:
            print(f"Skipping {file_path}: no valid data after date conversion")
            continue

        df.set_index('date', inplace=True)

        close_5min = df['close'].resample('5min').last()

        stock_name = file_path.replace('.csv', '').replace('./', '')
        close_price_data[stock_name] = close_5min

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

combined_close_prices = pd.DataFrame(close_price_data)

combined_close_prices = combined_close_prices.fillna(method='ffill')

price_returns = combined_close_prices.pct_change(fill_method=None).dropna()

stock_correlation_matrix = price_returns.corr()

print(f"Number of stocks processed: {len(close_price_data)}")
print(f"Date range: {combined_close_prices.index.min()} to {combined_close_prices.index.max()}")
print(f"Shape of combined data: {combined_close_prices.shape}")

print("\nCorrelation Matrix between Different Stocks (5-minute returns):")
print(stock_correlation_matrix.round(4))

correlation_pairs = []
for i in range(len(stock_correlation_matrix.columns)):
    for j in range(i+1, len(stock_correlation_matrix.columns)):
        stock1 = stock_correlation_matrix.columns[i]
        stock2 = stock_correlation_matrix.columns[j]
        correlation = stock_correlation_matrix.iloc[i, j]
        correlation_pairs.append((stock1, stock2, correlation))

correlation_pairs_sorted = sorted(correlation_pairs, key=lambda x: abs(x[2]), reverse=True)

print("\nTop 10 Most Correlated Stock Pairs:")
for i, (stock1, stock2, corr) in enumerate(correlation_pairs_sorted[:10]):
    print(f"{i+1}. {stock1} - {stock2}: {corr:.4f}")

high_correlation_pairs = [(s1, s2, c) for s1, s2, c in correlation_pairs if abs(c) > 0.7]

if high_correlation_pairs:
    print("\nHighly Correlated Stock Pairs (|correlation| > 0.7):")
    for stock1, stock2, corr in high_correlation_pairs:
        print(f"{stock1} - {stock2}: {corr:.4f}")
else:
    print("\nNo highly correlated stock pairs found (threshold: 0.7)")

print(f"\nCorrelation Summary:")
print(f"Average correlation: {stock_correlation_matrix.values[np.triu_indices_from(stock_correlation_matrix.values, k=1)].mean():.4f}")
print(f"Maximum correlation: {stock_correlation_matrix.values[np.triu_indices_from(stock_correlation_matrix.values, k=1)].max():.4f}")
print(f"Minimum correlation: {stock_correlation_matrix.values[np.triu_indices_from(stock_correlation_matrix.values, k=1)].min():.4f}")
