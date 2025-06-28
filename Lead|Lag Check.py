file_paths = glob.glob('*.csv')
close_price_data = {}

for file_path in file_paths:
    try:
        df = pd.read_csv(file_path)

        required_columns = ['date', 'close']
        if not all(col in df.columns for col in required_columns):
            continue

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)

        if df.empty:
            continue

        df.set_index('date', inplace=True)

        close_5min = df['close'].resample('5min').last()

        stock_name = file_path.split('/')[-1].replace('.csv', '')
        close_price_data[stock_name] = close_5min

    except Exception as e:
        continue

combined_close_prices = pd.DataFrame(close_price_data)
combined_close_prices = combined_close_prices.fillna(method='ffill')
price_returns = combined_close_prices.pct_change(fill_method=None).dropna()
stock_correlation_matrix = price_returns.corr()

correlation_pairs = []
for i in range(len(stock_correlation_matrix.columns)):
    for j in range(i+1, len(stock_correlation_matrix.columns)):
        stock1 = stock_correlation_matrix.columns[i]
        stock2 = stock_correlation_matrix.columns[j]
        correlation = stock_correlation_matrix.iloc[i, j]
        correlation_pairs.append((stock1, stock2, correlation))

correlation_pairs_sorted = sorted(correlation_pairs, key=lambda x: abs(x[2]), reverse=True)

print("Top 6 Most Correlated Stock Pairs:")
for i, (stock1, stock2, corr) in enumerate(correlation_pairs_sorted[:6]):
    print(f"{i+1}. {stock1} - {stock2}: {corr:.4f}")

def analyze_lead_lag_relationship(stock1, stock2, returns_data, max_lags=5):
    results = {}

    stock1_returns = returns_data[stock1].dropna()
    stock2_returns = returns_data[stock2].dropna()

    aligned_data = pd.DataFrame({
        'stock1': stock1_returns,
        'stock2': stock2_returns
    }).dropna()

    if len(aligned_data) < 50:
        return None

    for lag in range(1, max_lags + 1):
        stock1_lagged = aligned_data['stock1'].shift(lag)
        stock2_current = aligned_data['stock2']

        valid_data = pd.DataFrame({
            'stock1_lagged': stock1_lagged,
            'stock2_current': stock2_current
        }).dropna()

        if len(valid_data) > 20:
            correlation, p_value = pearsonr(valid_data['stock1_lagged'], valid_data['stock2_current'])
            results[f'{stock1}_leads_{stock2}_by_{lag}'] = {
                'correlation': correlation,
                'p_value': p_value,
                'sample_size': len(valid_data)
            }

        stock2_lagged = aligned_data['stock2'].shift(lag)
        stock1_current = aligned_data['stock1']

        valid_data = pd.DataFrame({
            'stock2_lagged': stock2_lagged,
            'stock1_current': stock1_current
        }).dropna()

        if len(valid_data) > 20:
            correlation, p_value = pearsonr(valid_data['stock2_lagged'], valid_data['stock1_current'])
            results[f'{stock2}_leads_{stock1}_by_{lag}'] = {
                'correlation': correlation,
                'p_value': p_value,
                'sample_size': len(valid_data)
            }

    return results

print("\nLead-Lag Analysis for Top 6 Correlated Pairs:")
print("=" * 60)

for i, (stock1, stock2, corr) in enumerate(correlation_pairs_sorted[:6]):
    print(f"\nPair {i+1}: {stock1} vs {stock2} (Correlation: {corr:.4f})")
    print("-" * 50)

    lead_lag_results = analyze_lead_lag_relationship(stock1, stock2, price_returns)

    if lead_lag_results is None:
        print("Insufficient data for lead-lag analysis")
        continue

    significant_relationships = []

    for relationship, stats in lead_lag_results.items():
        if stats['p_value'] < 0.05 and abs(stats['correlation']) > 0.1:
            significant_relationships.append((relationship, stats))

    if significant_relationships:
        significant_relationships.sort(key=lambda x: abs(x[1]['correlation']), reverse=True)

        print("Significant Lead-Lag Relationships (p < 0.05):")
        for relationship, stats in significant_relationships[:3]:
            parts = relationship.split('_')
            leader = parts[0]
            follower = parts[2]
            lag = parts[4]

            print(f"  {leader} leads {follower} by {lag} periods:")
            print(f"    Correlation: {stats['correlation']:.4f}")
            print(f"    P-value: {stats['p_value']:.4f}")
            print(f"    Sample size: {stats['sample_size']}")

            if abs(stats['correlation']) > 0.3:
                print(f"    ** STRONG LEADING INDICATOR **")
            elif abs(stats['correlation']) > 0.2:
                print(f"    ** MODERATE LEADING INDICATOR **")
            else:
                print(f"    ** WEAK LEADING INDICATOR **")
            print()
    else:
        print("No significant lead-lag relationships found")

def calculate_predictive_accuracy(stock1, stock2, returns_data, lag=1):
    aligned_data = pd.DataFrame({
        'leader': returns_data[stock1],
        'follower': returns_data[stock2]
    }).dropna()

    if len(aligned_data) < 50:
        return None

    leader_lagged = aligned_data['leader'].shift(lag)
    follower_current = aligned_data['follower']

    valid_data = pd.DataFrame({
        'leader_lagged': leader_lagged,
        'follower_current': follower_current
    }).dropna()

    if len(valid_data) < 30:
        return None

    leader_direction = (valid_data['leader_lagged'] > 0).astype(int)
    follower_direction = (valid_data['follower_current'] > 0).astype(int)

    accuracy = (leader_direction == follower_direction).mean()

    return accuracy

print("\nDirectional Predictive Accuracy Analysis:")
print("=" * 50)

for i, (stock1, stock2, corr) in enumerate(correlation_pairs_sorted[:6]):
    print(f"\nPair {i+1}: {stock1} vs {stock2}")

    accuracy_1_leads_2 = calculate_predictive_accuracy(stock1, stock2, price_returns, lag=1)
    accuracy_2_leads_1 = calculate_predictive_accuracy(stock2, stock1, price_returns, lag=1)

    if accuracy_1_leads_2 is not None:
        print(f"  {stock1} predicting {stock2} direction: {accuracy_1_leads_2:.3f} ({accuracy_1_leads_2*100:.1f}%)")
        if accuracy_1_leads_2 > 0.6:
            print(f"    ** {stock1} is a STRONG directional predictor of {stock2} **")
        elif accuracy_1_leads_2 > 0.55:
            print(f"    ** {stock1} is a MODERATE directional predictor of {stock2} **")

    if accuracy_2_leads_1 is not None:
        print(f"  {stock2} predicting {stock1} direction: {accuracy_2_leads_1:.3f} ({accuracy_2_leads_1*100:.1f}%)")
        if accuracy_2_leads_1 > 0.6:
            print(f"    ** {stock2} is a STRONG directional predictor of {stock1} **")
        elif accuracy_2_leads_1 > 0.55:
            print(f"    ** {stock2} is a MODERATE directional predictor of {stock1} **")
