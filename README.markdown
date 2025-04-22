# QuantumGapTrader: Advanced Equity Gap Trading Backtest Framework

**QuantumGapTrader** is a sophisticated Python-based backtesting engine for evaluating gap-based trading strategies on equity markets. Built on the `backtrader` framework, it executes trades triggered by significant price gaps (&gt;2%), implements circuit breaker logic (20% threshold), decomposes PnL across gap direction and trade type, supports customizable temporal scopes, and ranks top-performing and underperforming securities. The framework leverages Polars for high-performance data processing, employs sequential execution for robust stability, and provides real-time progress monitoring via `tqdm`. Outputs include granular trade logs, statistical analyses, visualizations, and a ranked performance CSV.

## Core Features

- **Gap-Based Trading Logic**:
  - **LONG**: Executes on gap-ups exceeding 2% (`Open > Previous Close * 1.02`).
  - **SHORT**: Triggers on gap-downs exceeding 2% (`Open < Previous Close * 0.98`).
  - **No Trade**: Skips gaps ≤2% or zero-gap scenarios (`Open == Previous Close`).
- **Circuit Breaker Filter**: Suppresses trades during upper/lower circuit events (e.g., `High == Close` and `Close ≥ Previous Close * 1.20`).
- **Scenario Decomposition**: Analyzes PnL across four quadrants:
  - Gap Up: Long/Short.
  - Gap Down: Long/Short.
- **Temporal Configurability**: Supports user-defined backtest periods (e.g., 2019-01-01 to 2025-12-31).
- **Performance Ranking**: Exports top 100 profitable and loss-making stocks to `stock_performance.csv`.
- **Visualization Suite**:
  - Per-security cumulative PnL plots.
  - Monthly PnL heatmaps.
  - Aggregate multi-security PnL plot.
- **Performance Optimization**:
  - Polars for accelerated DataFrame operations.
  - Sequential execution to mitigate multiprocessing instability (e.g., `BrokenProcessPool`).
  - `tqdm` for real-time progress visualization.
- **Profiling**: Generates `cProfile` output (`profile.prof`) for performance diagnostics.

## System Requirements

- **Python**: Version 3.8 or higher.
- **Git**: Required for version control and GitHub integration.
- **Data Input**: Daily OHLCV CSV files structured as:

  ```
  Date,Open,High,Low,Close,Volume
  2019-01-01T09:15:00+0530,100.0,105.0,98.0,102.0,10000
  ```
  - `Date` format: `YYYY-MM-DDTHH:MM:SS+0530`.
  - Store CSVs in a user-specified directory (default: `C:\Users\Yash\Desktop\Equity - Apr 2015 - Mar 2025\day`).

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/<your-username>/QuantumGapTrader.git
   cd QuantumGapTrader
   ```

2. **Establish a Virtual Environment** (strongly recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **Data Preparation**:

   - Place OHLCV CSV files in a designated directory.
   - Update `CONFIG['csv_directory']` in `src/quantum_gap_trader.py`:

     ```python
     'csv_directory': r'/path/to/your/csv/directory',
     ```

2. **Customize Parameters** (optional):

   - Modify `CONFIG` in `src/quantum_gap_trader.py`:

     ```python
     'start_date': '2019-01-01',        # Backtest start date
     'end_date': '2025-12-31',          # Backtest end date
     'circuit_limit': 0.20,             # Circuit breaker threshold (20%)
     'min_gap_pct': 2.0,                # Minimum gap percentage
     'capital_per_trade': 10000,        # Capital allocation per trade
     'performance_csv': 'stock_performance.csv'  # Output CSV
     ```

## Execution

1. **Run the Backtest**:

   ```bash
   python src/quantum_gap_trader.py
   ```

   - In Jupyter Notebook:

     ```python
     %run src/quantum_gap_trader.py
     ```

2. **Monitor Execution**:

   - A `tqdm` progress bar tracks CSV processing (e.g., `60%|██████ | 90/150 [02:18<01:32, 1.54s/file]`).
   - Logs provide detailed trade execution, statistical summaries, and top/bottom security rankings.

3. **Outputs**:

   - **Console Logs**:
     - Trade execution details.
     - Aggregate and scenario-based statistics (e.g., `Gap Up Long: Total PnL: ₹2000.00, Win Rate: 60.00%`).
     - Top 5 profitable/loss-making securities.
   - **Visualizations**:
     - `cumulative_return_<stock_name>.png`: Per-security cumulative PnL.
     - `monthly_pnl_heatmap_<stock_name>.png`: Monthly PnL heatmap.
     - `cumulative_return_all_stocks.png`: Aggregate PnL across all securities.
   - **Data Export**:
     - `stock_performance.csv`: Top 100 profitable/loss-making securities with `Stock,Total PnL`.
   - **Profiling**:
     - `profile.prof`: Performance diagnostics via `cProfile`.

4. **Analyze Results**:

   - Inspect scenario-based statistics for strategic insights.
   - Explore `stock_performance.csv`:

     ```bash
     head stock_performance.csv
     ```
   - Visualize plots using any image viewer.

## Example Output

```
Found 150 CSV files: ['0025_ADANIPORTS.csv', 'TCS.csv', ...]
Processing CSVs: 60%|██████    | 90/150 [02:18<01:32, 1.54s/file]
Processing 0025_ADANIPORTS...
2019-04-10: Gap Down: -2.50% (Open: 311.00, Previous Close: 319.00) | Short: 32 shares @ 311.00, Exit @ 308.00
Overall Summary Statistics for 0025_ADANIPORTS:
Total Trades: 50
Total PnL: ₹10000.00
Win Rate: 60.00%
Scenario-Based Statistics for 0025_ADANIPORTS:
Gap Up Long:
Total Trades: 10
Total PnL: ₹2000.00
Win Rate: 60.00%
...
Top Profitable Securities:
TCS: ₹15000.00
RELIANCE: ₹12000.00
...
Top Loss-Making Securities:
YESBANK: ₹-8000.00
IDEA: ₹-6000.00
...
Stock performance metrics exported to stock_performance.csv
```

## Diagnostic and Optimization Tools

1. **Profiling**:

   - Analyze performance bottlenecks:

     ```bash
     python -m pstats profile.prof
     ```
     - Commands: `sort cumulative`, `stats 10` (top 10 functions).
   - For line-level profiling:

     ```bash
     pip install line_profiler
     kernprof -l -v src/quantum_gap_trader.py
     ```
     - Add `@profile` decorator to functions like `GapStrategy.next`.

2. **Data Validation**:

   - Verify gap occurrences:

     ```python
     df = pl.read_csv('0025_ADANIPORTS.csv')
     df = df.with_columns(gap_pct=((pl.col('Open') - pl.col('Close').shift(1)) / pl.col('Close').shift(1) * 100))
     print(df.filter(pl.col('gap_pct').abs() > 2)[['Date', 'gap_pct']])
     ```

3. **Circuit Calibration**:

   - Adjust `circuit_limit` (e.g., `0.10` for 10% circuits).
   - Log circuit events:

     ```python
     if self.is_circuit(...):
         self.log(f"Circuit Event: High={high_price:.2f}, Low={low_price:.2f}, Close={close_price:.2f}")
     ```

## Troubleshooting

1. **No Trades Executed**:

   - Validate data for &gt;2% gaps (see above).
   - Confirm `start_date` and `end_date` align with CSV data.

2. **CSV Parsing Errors**:

   - Ensure CSVs adhere to the specified format.
   - Test a single CSV:

     ```python
     result = run_backtest_for_stock('/path/to/0025_ADANIPORTS.csv')
     print(result)
     ```

3. **Performance Bottlenecks**:

   - Inspect `profile.prof` for I/O or computation-heavy operations.
   - Optimize by reducing plot generation or using smaller datasets.

4. **Jupyter Compatibility**:

   - Restart kernel to mitigate profiling conflicts (`sys.setprofile(None)` included).

## Contribution Guidelines

- Fork the repository and create a feature branch:

  ```bash
  git checkout -b feature/enhancement
  ```
- Commit changes with descriptive messages:

  ```bash
  git commit -m "Implement enhanced gap size analysis"
  ```
- Push and submit a pull request:

  ```bash
  git push origin feature/enhancement
  ```

## License

Distributed under the MIT License. See `LICENSE` file (optional, to be added).

## Contact

For inquiries, bug reports, or feature requests, please open a GitHub issue or contact (optional).

---

*QuantumGapTrader: Precision-engineered for quantitative trading research.*