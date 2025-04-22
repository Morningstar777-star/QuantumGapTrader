import backtrader as bt
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os
from datetime import datetime
from typing import List, Tuple, Optional, Dict
import cProfile
import pstats
import logging
import sys
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'csv_directory': r'C:\Users\Yash\Desktop\Equity - Apr 2015 - Mar 2025\day',
    'capital_per_trade': 10000,
    'min_gap_pct': 2.0,
    'initial_cash': 1000000,
    'commission': 0.0,
    'date_format': '%Y-%m-%dT%H:%M:%S%z',
    'circuit_limit': 0.20,  # 20% circuit limit
    'start_date': '2019-01-01',
    'end_date': '2025-12-31',
    'plot_config': {
        'cumulative': {
            'figsize': (10, 6),
            'title': 'Cumulative Profit Over Time - {}',
            'xlabel': 'Date',
            'ylabel': 'Cumulative PnL (₹)',
            'color': 'blue',
            'filename': 'cumulative_return_{}.png'
        },
        'heatmap': {
            'figsize': (12, 8),
            'title': 'Monthly PnL Heatmap - {} (₹)',
            'xlabel': 'Month',
            'ylabel': 'Year',
            'cmap': 'RdYlGn',
            'filename': 'monthly_pnl_heatmap_{}.png'
        },
        'combined': {
            'figsize': (12, 8),
            'title': 'Cumulative Profit Over Time - All Stocks',
            'xlabel': 'Date',
            'ylabel': 'Cumulative PnL (₹)',
            'filename': 'cumulative_return_all_stocks.png'
        }
    },
    'performance_csv': 'stock_performance.csv'  # Output CSV for top/bottom stocks
}

class Plotter:
    @staticmethod
    def plot_cumulative_pnl(df: Optional[pl.DataFrame], name: str, config: Dict, combined: bool = False, all_dfs: Optional[List[pl.DataFrame]] = None, all_names: Optional[List[str]] = None):
        plt.figure(figsize=config['figsize'])
        if combined and all_dfs and all_names:
            for df_pnl, stock_name in zip(all_dfs, all_names):
                if df_pnl is not None and not df_pnl.is_empty():
                    plt.plot(df_pnl['Date'].to_pandas(), df_pnl['Cumulative PnL'], label=stock_name)
        elif df is not None and not df.is_empty():
            plt.plot(df['Date'].to_pandas(), df['Cumulative PnL'], label=f'Cumulative PnL ({name})', color=config['color'])
        plt.title(config['title'].format(name if not combined else 'All Stocks'))
        plt.xlabel(config['xlabel'])
        plt.ylabel(config['ylabel'])
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(config['filename'].format(name if not combined else 'all_stocks'))
        plt.close()

    @staticmethod
    def plot_monthly_heatmap(df: Optional[pl.DataFrame], name: str, config: Dict):
        if df is None or df.is_empty():
            return
        df = df.with_columns([
            pl.col('Date').dt.year().alias('Year'),
            pl.col('Date').dt.month().alias('Month')
        ])
        monthly_pnl = df.group_by(['Year', 'Month']).agg(pl.col('PnL').sum()).pivot(
            values='PnL', index='Year', columns='Month', aggregate_function='sum'
        ).fill_null(0)
        plt.figure(figsize=config['figsize'])
        sns.heatmap(monthly_pnl.to_pandas().iloc[:, 1:], annot=True, fmt='.2f', cmap=config['cmap'], center=0)
        plt.title(config['title'].format(name))
        plt.xlabel(config['xlabel'])
        plt.ylabel(config['ylabel'])
        plt.tight_layout()
        plt.savefig(config['filename'].format(name))
        plt.close()

class GapStrategy(bt.Strategy):
    params = (
        ('capital_per_trade', CONFIG['capital_per_trade']),
        ('min_gap_pct', CONFIG['min_gap_pct']),
        ('circuit_limit', CONFIG['circuit_limit']),
    )

    def __init__(self):
        self.trades = {
            'gap_up_long': [],
            'gap_up_short': [],
            'gap_down_long': [],
            'gap_down_short': []
        }
        self.previous_close = None

    def log(self, txt: str, dt: Optional[datetime.date] = None):
        dt = dt or self.datas[0].datetime.date(0)
        logger.info(f'{dt}: {txt}')

    def log_trade(self, date: datetime.date, trade_type: str, entry_price: float, exit_price: float, shares: int, pnl: float, gap_pct: float, gap_direction: str):
        trade_data = {
            'Date': date,
            'Type': trade_type,
            'Entry Price': entry_price,
            'Exit Price': exit_price,
            'Shares': shares,
            'PnL': pnl,
            'Gap %': gap_pct
        }
        key = f"gap_{gap_direction}_{trade_type.lower()}"
        self.trades[key].append(trade_data)

    def is_circuit(self, high: float, low: float, close: float, previous_close: float) -> bool:
        if previous_close == 0:
            return False
        upper_circuit = previous_close * (1 + self.params.circuit_limit)
        lower_circuit = previous_close * (1 - self.params.circuit_limit)
        is_upper = abs(high - close) < 1e-6 and close >= upper_circuit
        is_lower = abs(low - close) < 1e-6 and close <= lower_circuit
        return is_upper or is_lower

    def next(self):
        date = self.datas[0].datetime.date(0)
        open_price = self.datas[0].open[0]
        high_price = self.datas[0].high[0]
        low_price = self.datas[0].low[0]
        close_price = self.datas[0].close[0]

        if self.previous_close is None:
            self.previous_close = close_price
            return

        gap = open_price - self.previous_close
        gap_pct = (gap / self.previous_close * 100) if self.previous_close != 0 else 0
        abs_gap_pct = abs(gap_pct)
        gap_direction = 'up' if gap > 0 else 'down' if gap < 0 else 'zero'
        shares = int(self.params.capital_per_trade / open_price) if open_price > 0 else 0

        gap_info = f"Open: {open_price:.2f}, Previous Close: {self.previous_close:.2f}"

        # Check for circuit
        if self.is_circuit(high_price, low_price, close_price, self.previous_close):
            self.log(f"No trade: Circuit hit ({gap_info})")
            self.previous_close = close_price
            return

        if shares == 0:
            self.log(f"No trade: Insufficient capital ({gap_info})")
        elif abs_gap_pct <= self.params.min_gap_pct:
            direction = "Up" if gap > 0 else "Down" if gap < 0 else "Zero"
            self.log(f"Gap {direction}: {gap_pct:.2f}% (too small, >{self.params.min_gap_pct}% required, {gap_info})")
        elif open_price == self.previous_close:
            self.log(f"No trade: No gap ({gap_info})")
        else:
            trade_type = 'Long' if open_price > self.previous_close else 'Short'
            direction = 'Up' if trade_type == 'Long' else 'Down'
            self.log(f"Gap {direction}: {gap_pct:.2f}% ({gap_info}) | {trade_type}: {shares} shares @ {open_price:.2f}, Exit @ {close_price:.2f}")
            
            if trade_type == 'Long':
                self.buy(size=shares, price=open_price)
                self.sell(size=shares, price=close_price)
                pnl = (close_price - open_price) * shares
            else:
                self.sell(size=shares, price=open_price)
                self.buy(size=shares, price=close_price)
                pnl = (open_price - close_price) * shares
            
            self.log_trade(date, trade_type, open_price, close_price, shares, pnl, gap_pct, gap_direction)
            self.log(f"PnL: ₹{pnl:.2f}")

        self.previous_close = close_price

def analyze_stock(trades: Dict[str, List[Dict]], stock_name: str) -> Tuple[Optional[pl.DataFrame], float]:
    all_trades = []
    for key, trade_list in trades.items():
        all_trades.extend(trade_list)

    if not all_trades:
        logger.info(f"No trades executed for {stock_name}.")
        return None, 0.0

    try:
        df_trades = pl.DataFrame(all_trades)
        if df_trades.is_empty():
            return None, 0.0

        total_pnl = df_trades['PnL'].sum()
        df_trades = df_trades.with_columns([
            pl.col('Date').cast(pl.Datetime),
            pl.col('PnL').cum_sum().alias('Cumulative PnL')
        ])

        # Overall statistics
        wins = df_trades.filter(pl.col('PnL') > 0).shape[0]
        total_trades = df_trades.shape[0]
        summary = {
            'Total Trades': total_trades,
            'Wins': wins,
            'Losses': total_trades - wins,
            'Win Rate': (wins / total_trades * 100) if total_trades > 0 else 0,
            'Total PnL': total_pnl,
            'Average PnL': df_trades['PnL'].mean() if total_trades > 0 else 0
        }

        # Scenario-based statistics
        scenario_stats = {}
        for key in trades:
            df_scenario = pl.DataFrame(trades[key])
            if not df_scenario.is_empty():
                scenario_wins = df_scenario.filter(pl.col('PnL') > 0).shape[0]
                scenario_total = df_scenario.shape[0]
                scenario_stats[key] = {
                    'Total Trades': scenario_total,
                    'Wins': scenario_wins,
                    'Losses': scenario_total - scenario_wins,
                    'Win Rate': (scenario_wins / scenario_total * 100) if scenario_total > 0 else 0,
                    'Total PnL': df_scenario['PnL'].sum(),
                    'Average PnL': df_scenario['PnL'].mean() if scenario_total > 0 else 0
                }

        logger.info(f"Trade Log for {stock_name}:\n{df_trades[['Date', 'Type', 'Entry Price', 'Exit Price', 'Shares', 'Gap %', 'PnL']].to_pandas().to_string(index=False)}")
        logger.info(f"\nOverall Summary Statistics for {stock_name}:")
        for key, value in summary.items():
            logger.info(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
        
        logger.info(f"\nScenario-Based Statistics for {stock_name}:")
        for scenario, stats in scenario_stats.items():
            logger.info(f"\n{scenario.replace('_', ' ').title()}:")
            for key, value in stats.items():
                logger.info(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")

        Plotter.plot_cumulative_pnl(df_trades, stock_name, CONFIG['plot_config']['cumulative'])
        Plotter.plot_monthly_heatmap(df_trades, stock_name, CONFIG['plot_config']['heatmap'])

        return df_trades[['Date', 'Cumulative PnL']], total_pnl
    except Exception as e:
        logger.error(f"Error analyzing stock {stock_name}: {e}")
        return None, 0.0

def run_backtest_for_stock(csv_file: str, config: Dict = CONFIG) -> Tuple[Optional[pl.DataFrame], str, float]:
    stock_name = os.path.splitext(os.path.basename(csv_file))[0]
    logger.info(f"Processing {stock_name}...")

    try:
        cerebro = bt.Cerebro()
        cerebro.addstrategy(GapStrategy)
        cerebro.broker.setcash(config['initial_cash'])
        cerebro.broker.setcommission(commission=config['commission'])

        start_date = datetime.strptime(config['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(config['end_date'], '%Y-%m-%d')

        data = bt.feeds.GenericCSVData(
            dataname=csv_file,
            fromdate=start_date,
            todate=end_date,
            nullvalue=0.0,
            dtformat=config['date_format'],
            datetime=0, open=1, high=2, low=3, close=4, volume=5, openinterest=-1
        )
        cerebro.adddata(data)

        strats = cerebro.run()
        if strats:
            df_trades, total_pnl = analyze_stock(strats[0].trades, stock_name)
            return df_trades, stock_name, total_pnl
        logger.error(f"Backtest failed for {stock_name}.")
        return None, stock_name, 0.0
    except Exception as e:
        logger.error(f"Error in backtest for {csv_file}: {e}")
        return None, stock_name, 0.0

def export_stock_performance(stock_pnls: List[Tuple[str, float]], config: Dict):
    try:
        df = pl.DataFrame({
            'Stock': [name for name, _ in stock_pnls],
            'Total PnL': [pnl for _, pnl in stock_pnls]
        })
        
        # Top 100 profitable stocks
        profitable = df.filter(pl.col('Total PnL') > 0).sort('Total PnL', descending=True).head(100)
        # Top 100 loss-making stocks
        loss_making = df.filter(pl.col('Total PnL') < 0).sort('Total PnL', descending=False).head(100)
        
        # Combine and save
        combined = pl.concat([profitable, loss_making])
        combined.write_csv(config['performance_csv'])
        logger.info(f"Stock performance saved to {config['performance_csv']}")
        
        # Log top/bottom stocks
        logger.info("\nTop Profitable Stocks:")
        for row in profitable.head(5).rows(named=True):
            logger.info(f"{row['Stock']}: ₹{row['Total PnL']:.2f}")
        logger.info("\nTop Loss-Making Stocks:")
        for row in loss_making.head(5).rows(named=True):
            logger.info(f"{row['Stock']}: ₹{row['Total PnL']:.2f}")
    except Exception as e:
        logger.error(f"Error exporting stock performance: {e}")

def run_multi_stock_backtest(csv_files: List[str], config: Dict = CONFIG) -> None:
    if not csv_files:
        logger.error(f"No CSV files found in {config['csv_directory']}.")
        logger.info(f"Current directory contents: {os.listdir(config['csv_directory'])}")
        return

    logger.info(f"Found {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")
    
    all_pnl_dfs = []
    stock_names = []
    stock_pnls = []

    # Sequential execution with tqdm progress bar
    results = []
    for csv_file in tqdm(csv_files, desc="Processing CSVs", unit="file"):
        result = run_backtest_for_stock(csv_file, config)
        results.append(result)

    for df_trades, stock_name, total_pnl in results:
        if df_trades is not None:
            all_pnl_dfs.append(df_trades)
            stock_names.append(stock_name)
        stock_pnls.append((stock_name, total_pnl))

    if all_pnl_dfs:
        Plotter.plot_cumulative_pnl(None, 'all_stocks', config['plot_config']['combined'], combined=True, all_dfs=all_pnl_dfs, all_names=stock_names)
    else:
        logger.info("No valid results to plot combined PnL.")

    # Export stock performance
    export_stock_performance(stock_pnls, config)

def main():
    sys.setprofile(None)
    csv_files = glob.glob(os.path.join(CONFIG['csv_directory'], '*.csv'))
    profiler = cProfile.Profile()
    try:
        profiler.enable()
        run_multi_stock_backtest(csv_files)
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise
    finally:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.dump_stats('profile.prof')
        stats.print_stats()

if __name__ == '__main__':
    main()