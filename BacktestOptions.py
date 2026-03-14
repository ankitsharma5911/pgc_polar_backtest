import os
import gc
import math
import time
import datetime
import requests
import traceback
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from time import sleep
import concurrent.futures
from functools import lru_cache
pd.set_option('future.no_silent_downcasting', True)

NSE_INDICES = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']
BSE_INDICES = ['BANKEX', 'SENSEX']
MCX_INDICES = ['CRUDEOIL', 'CRUDEOILM', 'NATGASMINI', 'NATURALGAS', 'COPPER', 'SILVER', 'GOLD', 'SILVERM', 'GOLDM', 'ZINC']
US_INDICES = ['AAPL', 'AMD', 'AMZN', 'BABA', 'GOOGL', 'HOOD', 'INTC', 'MARA', 'META', 'MSFT', 'MSTR', 'NVDA', 'PLTR', 'QQQ_FRI', 'QQQ_MON', 'QQQ_THU', 'QQQ_TUE', 'QQQ_WED', 'SMCI', 'SOFI', 'SPXW_FRI', 'SPXW_MON', 'SPXW_THU', 'SPXW_TUE', 'SPXW_WED', 'SPY_FRI', 'SPY_MON', 'SPY_THU', 'SPY_TUE', 'SPY_WED', 'TSLA', 'UVIX', 'UVXY', 'VIX', 'VXX', 'XSP_FRI', 'XSP_MON', 'XSP_THU', 'XSP_TUE', 'XSP_WED']

class DataEmptyError(Exception):
    pass

def get_pm_time_index(dates, meta_start_time, meta_end_time):
    
    if isinstance(dates, (list, tuple, pd.Series)):
        time_index = pd.DatetimeIndex([])
        for date in dates:
            daily_index = pd.date_range(
                start=datetime.datetime.combine(date, meta_start_time),
                end=datetime.datetime.combine(date, meta_end_time),
                freq='1min'
            )
            time_index = time_index.append(daily_index)
    else:
        date = dates
        time_index = pd.date_range(datetime.datetime.combine(date, meta_start_time), datetime.datetime.combine(date, meta_end_time), freq='1min')
    
    return time_index

def set_pm_time_index(data, time_index):
    if data.empty:
        return pd.Series(0, index=time_index)
    return data.reindex(index=time_index, method='ffill', fill_value=0, copy=True)

cv = lambda x: str(float(x)) if isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '', 1).isdigit()) else x

def cal_percent(price, percent):
    return price * percent/100

def get_strike(scrip):
    strike = float(scrip[:-2])
    strike = int(strike) if strike.is_integer() else strike
    return strike

chunk_size = 100_000
def is_file_exists(output_csv_path, file_name, parameter_size, dir_files=None, cache=False):
    
    total_chunks = (parameter_size - 1) // chunk_size + 1

    if cache:
        if not hasattr(is_file_exists, '_cached_dir_files'):
            is_file_exists._cached_dir_files = set(os.listdir(output_csv_path)) if os.path.exists(output_csv_path) else set()
            dir_files = is_file_exists._cached_dir_files
        else:
            dir_files = is_file_exists._cached_dir_files if dir_files is None else dir_files

    if dir_files is None:
        return all(os.path.exists(f"{output_csv_path}{file_name} No-{idx}.parquet") for idx in range(1, total_chunks + 1))
    else:
        return all(f"{file_name} No-{idx}.parquet" in dir_files for idx in range(1, total_chunks + 1))

def save_chunk_data(chunk, log_cols, chunk_file_name):
    chunk = [d for d in chunk if d is not None]
    log_data_chunk = pd.DataFrame(chunk, columns=log_cols)
    log_data_chunk.replace('', np.nan, inplace=True)
    
    dir_path = os.path.dirname(chunk_file_name)
    while True:
        try:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Directory {dir_path} not available")

            log_data_chunk.to_parquet(chunk_file_name, index=False)
            return
        except Exception as e:
            print(f"Save failed ({e}), retrying in {5}s...")
            time.sleep(5)

class IntradayBacktest:
    
    PREFIX = {'nifty': 'Nifty', 'banknifty': 'BN', 'finnifty': 'FN', 'midcpnifty': 'MCN', 'sensex': 'SX','bankex': 'BX'}
    SLIPAGES = {'nifty': 0.01, 'banknifty': 0.0125, 'finnifty': 0.01, 'midcpnifty': 0.0125, 'sensex': 0.0125, 'bankex': 0.0125}
    STEPS = {'nifty': 1000, 'banknifty': 5000, 'finnifty': 1000, 'midcpnifty': 1000, 'sensex': 5000,'bankex': 5000, 'spxw': 500, 'xsp': 50}
    STEPS.update({'crudeoil':500, 'crudeoilm':500, 'natgasmini':50, 'naturalgas':50})
    TICKS = {'crudeoil':0.10} # Except all 0.05
    
    ROUNDING_MAP = {
        ('SELL', 'STOPLOSS'): math.ceil,
        ('SELL', 'TARGET'): math.floor,
        ('SELL', 'DECAY'): math.floor,
        ('BUY', 'STOPLOSS'): math.floor,
        ('BUY', 'TARGET'): math.ceil,
        ('BUY', 'DECAY'): math.ceil,
    }

    token, group_id = '5156026417:AAExQbrMAPrV0qI8tSYplFDjZltLBzXTm1w', '-607631145'

    def __init__(self, pickle_path, index, current_date, dte, start_time, end_time):
        
        self.pickle_path, self.index, self.current_date, self.dte, self.meta_start_time, self.meta_end_time = pickle_path, index, current_date, dte, start_time, end_time
        self.__future_pickle_path, self.__option_pickle_path = self.get_future_option_path(index)

        future_parquet_path = self.__future_pickle_path.format(date=self.current_date.date(), extn='parquet')
        future_pickle_path = self.__future_pickle_path.format(date=self.current_date.date(), extn='pkl')

        if os.path.exists(future_parquet_path):
            self.future_data = pl.read_parquet(future_parquet_path)
        elif os.path.exists(future_pickle_path):
            self.future_data = pl.from_pandas(pd.read_pickle(future_pickle_path))
        else:
            raise FileNotFoundError(f"Future data file not found for {self.index} on {self.current_date.date()}")
            
        self.future_data_pl = self.future_data.select(["date_time", "open", "high", "low", "close"]).with_columns(pl.col("date_time").cast(pl.Datetime))
        option_parquet_path = self.__option_pickle_path.format(date=self.current_date.date(), extn='parquet')
        option_pickle_path = self.__option_pickle_path.format(date=self.current_date.date(), extn='pkl')
        
        if os.path.exists(option_parquet_path):
            self.options_pl = pl.read_parquet(option_parquet_path)
        elif os.path.exists(option_pickle_path):
            self.options_pl = pl.from_pandas(pd.read_pickle(option_pickle_path))
        else:
            raise FileNotFoundError(f"Option data file not found for {self.index} on {self.current_date.date()}")
        
        self.options_pl = self.options_pl.select(["scrip", "date_time", "open", "high", "low", "close"])
        self.options_pl = self.options_pl.with_columns(pl.col("date_time").cast(pl.Datetime))
        self.options_pl = self.options_pl.filter((pl.col("date_time").dt.time() >= self.meta_start_time) & (pl.col("date_time").dt.time() <= self.meta_end_time))
        self.options_pl = self.options_pl  # Polars version for efficient filtering
        self.gap = self.get_gap()
        self.tick_size = self.TICKS.get(self.index.lower(), 0.05)
        
        if self.index in NSE_INDICES:
            self.market = 'NSE'
        elif self.index in BSE_INDICES:
            self.market = 'BSE'
        elif self.index in MCX_INDICES:
            self.market = 'MCX'
        elif self.index in US_INDICES:
            self.market = 'US'
        else:
            self.market = 'OTHER'

        self.get_single_leg_data = lru_cache(maxsize=4096)(self._get_single_leg_data)
        self.get_straddle_data = lru_cache(maxsize=4096)(self._get_straddle_data)
        self.get_strike = lru_cache(maxsize=4096)(self._get_strike)
        self.sl_check_single_leg = lru_cache(maxsize=4096)(self._sl_check_single_leg)
        self.sl_check_combine_leg = lru_cache(maxsize=4096)(self._sl_check_combine_leg)
        self.decay_check_single_leg = lru_cache(maxsize=4096)(self._decay_check_single_leg)
        self.sl_check_single_leg_with_sl_trail = lru_cache(maxsize=4096)(self._sl_check_single_leg_with_sl_trail)
        self.sl_check_combine_leg_with_sl_trail = lru_cache(maxsize=4096)(self._sl_check_combine_leg_with_sl_trail)
        self.straddle_indicator = lru_cache(maxsize=4096)(self._straddle_indicator)
        self.get_option_close = lru_cache(maxsize=1024)(self._get_option_close)

    def get_future_option_path(self, index):
        index_lower = index.lower()
        future_pickle_path = f'{self.pickle_path}{self.PREFIX.get(index_lower, index)} Future/{{date}}_{index_lower}_future.{{extn}}'
        option_pickle_path = f'{self.pickle_path}{self.PREFIX.get(index_lower, index)} Options/{{date}}_{index_lower}.{{extn}}'
        return future_pickle_path, option_pickle_path

    def Cal_slipage(self, price):
        return price * self.SLIPAGES.get(self.index.lower(), 0.01)
    
    def send_tg_msg(self, msg):
        print(msg)
        try:
            requests.get(f'https://api.telegram.org/bot{self.token}/sendMessage?chat_id={self.group_id}&text={msg}')
        except:
            pass

    def get_gap(self):
        try:
            strike = self.options_pl.select(pl.col("scrip")).unique().to_series().to_list()
            strike = [float(x[:-2]) for x in strike]
            strike = list(sorted(set(strike)))
            differences = np.diff(strike)
            min_gap = float(differences.min())
            min_gap = int(min_gap) if min_gap.is_integer() else min_gap
            return min_gap
        except Exception as e:
            print(e)

    def get_one_om(self, future_price=None):

        future_price = self.future_data['close'].iloc[0] if future_price is None else future_price

        if self.index.lower() in self.STEPS:
            step = self.STEPS[self.index.lower()]
            return ((int(future_price/step)*step)/100)
        else:
            return cal_percent(round(future_price), 1)
        
    def _get_option_close(self, dt, scrip):
        df = self.options_pl.filter(
            (pl.col("date_time") == dt) &
            (pl.col("scrip") == scrip)
        ).select("close")

        if df.is_empty():
            return None
        return df.item()

    def round_to_ticksize(self, value, orderside, ordertype): 
        round_func = self.ROUNDING_MAP[(orderside, ordertype)]
        return round(round_func(value / self.tick_size) * self.tick_size, 2)

    def _get_single_leg_data(self, start_dt, end_dt, scrip, pandas=False):
        # Filter using Polars for better performance
        data = self.options_pl.filter(
            (pl.col('scrip') == scrip) &
            (pl.col('date_time') >= start_dt) &
            (pl.col('date_time') <= end_dt)
        )
        if pandas:
            return data.to_pandas()
        else:
            return data

    def _get_straddle_data(self, start_dt, end_dt, ce_scrip, pe_scrip, seperate=False, pandas=False):

        ce_data:pl.DataFrame = self.get_single_leg_data(start_dt, end_dt, ce_scrip)
        pe_data:pl.DataFrame = self.get_single_leg_data(start_dt, end_dt, pe_scrip)
        
        if ce_data.is_empty() or pe_data.is_empty():
            if seperate:
                return  (pd.DataFrame(), pd.DataFrame()) if pandas else (pl.DataFrame(), pl.DataFrame())
            return (pd.DataFrame(), pd.DataFrame()) if pandas else (pl.DataFrame(), pl.DataFrame())
        
        # Get common date_times using semi join
        common_times = ce_data.select('date_time').join(
            pe_data.select('date_time'), on='date_time', how='inner'
        )
        
        # Filter both dataframes to only include common times
        ce_data = ce_data.filter(pl.col('date_time').is_in(common_times['date_time'])).sort('date_time')
        pe_data = pe_data.filter(pl.col('date_time').is_in(common_times['date_time'])).sort('date_time')
        
        if seperate:
            return ce_data.to_pandas(), pe_data.to_pandas() if pandas else (ce_data, pe_data)
        else:
            # Calculate straddle data using Polars expressions
            straddle_data = pl.DataFrame({
                'date_time': ce_data['date_time'],
            }).with_columns([
                pl.max_horizontal(
                    ce_data['high'] + pe_data['low'],
                    ce_data['low'] + pe_data['high']
                ).alias('high'),
                pl.min_horizontal(
                    ce_data['high'] + pe_data['low'],
                    ce_data['low'] + pe_data['high']
                ).alias('low'),
                (ce_data['close'] + pe_data['close']).alias('close')
            ])
            
            return straddle_data.to_pandas() if pandas else straddle_data

    def get_straddle_strike(self, start_dt, end_dt, sd=0, SDroundoff=False):

        future_pl = (
            self.future_data_pl.filter(
                (pl.col("date_time") >= start_dt) &
                (pl.col("date_time") <= end_dt)
            )
            .select(["date_time", "close"])
            .sort("date_time")
        )
        if future_pl.is_empty():
            return (None,) * 6
        
        for row in future_pl.iter_rows(named=True):
            try:
                current_dt = row["date_time"]
                future_price = row["close"]
                round_future_price = round(future_price/self.gap)*self.gap
                
                ce_scrip, pe_scrip = f"{round_future_price}CE", f"{round_future_price}PE"
                ce_price = self.get_option_close(current_dt, ce_scrip)
                pe_price = self.get_option_close(current_dt, pe_scrip)

                if ce_price is None or pe_price is None:
                    continue
                
                # Synthetic future
                syn_future = ce_price - pe_price + round_future_price
                round_syn_future = round(syn_future/self.gap)*self.gap
                
                # Scrip lists
                ce_scrip_list = [f"{round_syn_future}CE", f"{round_syn_future+self.gap}CE", f"{round_syn_future-self.gap}CE"]
                pe_scrip_list = [f"{round_syn_future}PE", f"{round_syn_future+self.gap}PE", f"{round_syn_future-self.gap}PE"]
                
                selected, min_value = None, float("inf")
                for i in range(3):
                    try:
                        ce_price = self.get_option_close(current_dt, ce_scrip_list[i])
                        pe_price = self.get_option_close(current_dt, pe_scrip_list[i])
                        
                        if ce_price is None or pe_price is None: continue
                            
                        diff = abs(ce_price-pe_price)
                        if min_value > diff:
                            min_value = diff
                            selected = (ce_scrip_list[i], pe_scrip_list[i], ce_price, pe_price)
                    except:
                        pass
                
                if selected is None: continue
                ce_scrip, pe_scrip, ce_price, pe_price = selected
                if sd:
                    sd_range = (ce_price + pe_price) * sd
                    
                    if SDroundoff:
                        sd_range = round(sd_range/self.gap)*self.gap
                    else:
                        sd_range = max(self.gap, round(sd_range/self.gap)*self.gap)
                    
                    ce_scrip, pe_scrip = f"{get_strike(ce_scrip) + sd_range}CE", f"{get_strike(pe_scrip) - sd_range}PE"
                    ce_price, pe_price = self.get_option_close(current_dt, ce_scrip), self.get_option_close(current_dt, pe_scrip)

                    if ce_price is None or pe_price is None:
                        continue
                    
                return ce_scrip, pe_scrip, ce_price, pe_price, future_price, current_dt
            except (IndexError, KeyError, ValueError, TypeError):
                continue
            except Exception as e:
                print('get_straddle_strike', e)
                traceback.print_exc()
                continue
        return (None,) * 6
        
    def get_strangle_strike(self, start_dt, end_dt, om=None, target=None, check_inverted=False, tf=1):

        future_pl = (
            self.future_data_pl.filter(
                (pl.col("date_time") >= start_dt) &
                (pl.col("date_time") <= end_dt)
            ).select(["date_time", "close"]).sort("date_time"))
        
        for row in future_pl.iter_rows(named=True):
            try:
                current_dt = row["date_time"]
                future_price = row["close"]
                one_om = self.get_one_om(future_price)
                target = one_om * om if target is None else target
                target_od = (
                    self.options_pl.filter(
                        (pl.col("date_time") == current_dt) &
                        (pl.col("close") >= target * tf)
                    ).sort("close"))
                
                ce_scrip = (
                    target_od.filter(pl.col("scrip").cast(pl.Utf8).str.ends_with("CE"))
                    .select("scrip")
                    .item(0)
                )
                pe_scrip = (
                    target_od.filter(pl.col("scrip").cast(pl.Utf8).str.ends_with("PE"))
                    .select("scrip")
                    .item(0)
                )
                
                ce_scrip_list = [ce_scrip, f"{get_strike(ce_scrip) - self.gap}CE", f"{get_strike(ce_scrip) + self.gap}CE"]
                pe_scrip_list = [pe_scrip, f"{get_strike(pe_scrip) - self.gap}PE", f"{get_strike(pe_scrip) + self.gap}PE"]

                call_list_prices, put_list_prices = [], []
                
                for z in range(3):
                    try:
                        call_list_prices.append(
                            self.get_option_close(current_dt, ce_scrip_list[z]))
                    except:
                        call_list_prices.append(0)
                    try:
                        put_list_prices.append(
                            self.get_option_close(current_dt, pe_scrip_list[z]))
                    except:
                        put_list_prices.append(0)
                
                call, put = call_list_prices[0], put_list_prices[0]
                
                min_diff = float("inf")
                target_2 = target * 2 * tf
                target_3 = target * 3
                
                required_call, required_put = None, None
                diff = abs(put - call)
                if (put+call >= target_2) and (min_diff > diff) and (put+call <= target_3):
                    min_diff = diff
                    required_call, required_put = call, put
                for i in range(1, 3):
                    if (abs(put_list_prices[i] - call) < min_diff) and (target_2 <= (put_list_prices[i] + call) <= target_3):
                        min_diff = abs(put_list_prices[i] - call)
                        required_call, required_put = call, put_list_prices[i]

                    if (abs(call_list_prices[i] - put) < min_diff) and (target_2 <= (call_list_prices[i] + put) <= target_3):
                        min_diff = abs(call_list_prices[i] - put)
                        required_call, required_put = call_list_prices[i], put
                
                if required_call is None or required_put is None: continue
                ce_scrip = ce_scrip_list[call_list_prices.index(required_call)]
                pe_scrip = pe_scrip_list[put_list_prices.index(required_put)]
                
                ce_price, pe_price = self.get_option_close(current_dt, ce_scrip), self.get_option_close(current_dt, pe_scrip)
                
                if get_strike(ce_scrip) < get_strike(pe_scrip) and check_inverted:
                    return self.get_straddle_strike(current_dt, end_dt)
                
                return ce_scrip, pe_scrip, ce_price, pe_price, future_price, current_dt
            except (IndexError, KeyError, ValueError, TypeError):
                continue
            except Exception as e:
                print('get_straddle_strike', e)
                traceback.print_exc()
                continue
        return None, None, None, None, None, None
                 
    def get_ut_strike(self, start_dt, end_dt, om=None, target=None):

        future_df = (self.future_data_pl.filter(
            (pl.col("date_time") >= start_dt) &
            (pl.col("date_time") <= end_dt)
        ).select(["date_time", "close"]).sort("date_time"))
        
        if future_df.is_empty(): return (None,) * 6
            
        for row in future_df.iter_rows(named=True):
            try:
                current_dt = row["date_time"]
                future_price = row["close"]
                one_om = self.get_one_om(future_price)
                tgt = one_om * om if target is None else target
                
                target_od = (self.options_pl.filter(pl.col("date_time") == current_dt).select(["scrip", "close"]))
                if target_od.is_empty(): continue
                
                ce_df = (target_od.filter((pl.col("scrip").str.ends_with("CE")) & (pl.col("close") >= tgt)).sort("close"))
                pe_df = (target_od.filter((pl.col("scrip").str.ends_with("PE")) & (pl.col("close") >= tgt)).sort("close"))
                
                if ce_df.is_empty() or pe_df.is_empty(): continue
                
                ce_scrip, ce_price = ce_df.row(0)
                pe_scrip, pe_price = pe_df.row(0)
                
                return ce_scrip, pe_scrip, ce_price, pe_price, future_price, current_dt
            except (IndexError, KeyError, ValueError, TypeError):
                    continue
            except Exception as e:
                print('get_straddle_strike', e)
                traceback.print_exc()
                continue
                
        return (None,) * 6

    def _get_strike(self, start_dt, end_dt, om=None, target=None, check_inverted=False, tf=1, only=None, obove_target_only=False, SDroundoff=False):
        
        if '%' in str(om) or obove_target_only:
            
            if '%' in str(om):
                om_precent = float(om.replace('%', ''))
                future_price = self.future_data_pl.select('close').row(0)[0]
                one_om = self.get_one_om(future_price)
                target = one_om*om_precent/100

            ce_scrip, pe_scrip, ce_price, pe_price, future_price, start_dt = self.get_ut_strike(start_dt, end_dt, om=om, target=target)  
        else:
            if 'SD' in str(om).upper() and om is not None:
                sd = float(om.upper().replace(' ', '').replace('SD', ''))
                om = None
            else:
                sd = 0
                om = float(om) if om else om

            if (om is None or om <= 0) and target is None:
                ce_scrip, pe_scrip, ce_price, pe_price, future_price, start_dt = self.get_straddle_strike(start_dt, end_dt, sd=sd, SDroundoff=SDroundoff)
            else:
                ce_scrip, pe_scrip, ce_price, pe_price, future_price, start_dt = self.get_strangle_strike(start_dt, end_dt, om=om, target=target, check_inverted=check_inverted, tf=tf)
                
        if only is None:
            return ce_scrip, pe_scrip, ce_price, pe_price, future_price, start_dt
        else:
            if only == "CE":
                return ce_scrip, ce_price, future_price, start_dt
            elif only == "PE":
                return pe_scrip, pe_price, future_price, start_dt
            return None, None, None, None
            
    def sl_check_by_given_data(self, scrip_df, o=None, sl=0, intra_sl=0, sl_price=None, target_price=None, from_candle_close=False, orderside='SELL', from_next_minute=True, with_ohlc=False, pl_with_slipage=True, per_minute_mtm=False, roundtick=False,pandas=True):
        """
        # Pandas in → Pandas out (default)
        sl_check_by_given_data(df)

        # Polars in → Polars out
        sl_check_by_given_data(pl_df, pandas=False)

        # Pandas with MTM
        sl_check_by_given_data(df, per_minute_mtm=True)

        # Polars MTM (ultra fast)
        sl_check_by_given_data(pl_df, per_minute_mtm=True, pandas=False)

        """
        sl_flag, intra_sl_flag, target_flag, exit_time, pnl = False, False, False, '', 0
        try:
            if pandas:
                if scrip_df.empty: raise DataEmptyError
                pl_df = pl.from_pandas(scrip_df.reset_index())
            else:
                if scrip_df.is_empty(): raise DataEmptyError
                pl_df = scrip_df
            
            o = pl_df.select("close").item(0) if o is None else o
            slipage = self.Cal_slipage(o) if pl_with_slipage else 0

            if from_next_minute: pl_df = pl_df.slice(1)
            if pl_df.is_empty(): raise DataEmptyError

            h = pl_df.select(pl.col("high").max()).item()
            l = pl_df.select(pl.col("low").min()).item()
            c = pl_df.select(pl.col("close").last()).item()
            
            if orderside == 'SELL':
                sl_price_val = (((100 + sl) / 100) * o if sl_price is None else sl_price) if (sl or sl_price) else (h + 1)
                intra_sl_price = ((100 + intra_sl) / 100) * o if intra_sl else (h + 1)
                target_price = target_price if target_price is not None else (l - 1)
            else:
                sl_price_val = (((100 - sl) / 100) * o if sl_price is None else sl_price) if (sl or sl_price) else (l - 1)
                intra_sl_price = ((100 - intra_sl) / 100) * o if intra_sl else (l - 1)
                target_price = target_price if target_price is not None else (h + 1)

            if roundtick or self.market == 'MCX':
                sl_price_val = self.round_to_ticksize(sl_price_val, orderside, 'STOPLOSS')
                intra_sl_price = self.round_to_ticksize(intra_sl_price, orderside, 'STOPLOSS')
                target_price = self.round_to_ticksize(target_price, orderside, 'TARGET')

            if orderside == 'SELL':
                sl_col = "close" if from_candle_close else "high"
                exit_cond = (
                    (pl.col("high") >= intra_sl_price) |
                    (pl.col(sl_col) >= sl_price_val) |
                    (pl.col("low") <= target_price))
            else:
                sl_col = "close" if from_candle_close else "low"
                exit_cond = (
                    (pl.col("low") <= intra_sl_price) |
                    (pl.col(sl_col) <= sl_price_val) |
                    (pl.col("high") >= target_price))
            
            combined_mask = pl_df.filter(exit_cond).sort("date_time")

            if not combined_mask.is_empty():
                exit_row = combined_mask.row(0, named=True)
                exit_time = exit_row["date_time"]


                if orderside == 'SELL':
                    if exit_row['high'] >= intra_sl_price:
                        sl_flag, intra_sl_flag = True, True
                        exit_price = intra_sl_price
                    elif (exit_row['close'] if from_candle_close else exit_row['high']) >= sl_price_val:
                        sl_flag = True
                        exit_price = exit_row['close'] if from_candle_close else sl_price_val 
                    elif exit_row['low'] <= target_price:
                        target_flag = True
                        exit_price = target_price
                elif orderside == 'BUY':
                    if exit_row['low'] <= intra_sl_price:
                        sl_flag, intra_sl_flag = True, True
                        exit_price = intra_sl_price
                    elif (exit_row['close'] if from_candle_close else exit_row['low']) <= sl_price_val:
                        sl_flag = True
                        exit_price = exit_row['close'] if from_candle_close else sl_price_val
                    elif exit_row['high'] >= target_price:
                        target_flag = True
                        exit_price = target_price
            else:
                exit_price = c

            pnl = (exit_price - o) if orderside == 'BUY' else (o - exit_price)
            pnl = round(pnl - slipage, 2)
            
            mtm = None
            if per_minute_mtm:
                scrip_df = scrip_df.to_pandas()
                scrip_df.set_index('date_time', inplace=True)
                if exit_time:
                    scrip_df = scrip_df.loc[scrip_df.index <= exit_time]

                per_minute_mtm_series = o - scrip_df['close'] if orderside == 'SELL' else scrip_df['close'] - o
                per_minute_mtm_series = per_minute_mtm_series - slipage
                per_minute_mtm_series.iloc[-1] = pnl
                if not pandas:
                    per_minute_mtm_series = pl.Series(per_minute_mtm_series)
            
                

        except DataEmptyError:
            sl_flag, intra_sl_flag, target_flag, exit_time, pnl = False, False, False, '', 0
            o, h, l, c = '', '', '', ''
            per_minute_mtm_series = pd.Series() if pandas else pl.Series()
            sl_price_val = ''
        except Exception as e:
            print('sl_check_single_leg', e)
            traceback.print_exc()
            sl_flag, intra_sl_flag, target_flag, exit_time, pnl = False, False, False, '', 0
            o, h, l, c = '', '', '', ''
            per_minute_mtm_series = pd.Series() if pandas else pl.Series()
            sl_price_val = ''

        sl_price = sl_price_val if (sl or sl_price) else ''

        if with_ohlc:
            ohlc_data = (o, h, l, c, sl_price)
            if per_minute_mtm:
                return (*ohlc_data, exit_time, per_minute_mtm_series)
            return (*ohlc_data, sl_flag, intra_sl_flag, target_flag, exit_time, pnl)
        if per_minute_mtm:
            return (exit_time, per_minute_mtm_series)
        return (sl_price, sl_flag, intra_sl_flag, target_flag, exit_time, pnl)

    def _sl_check_single_leg(self, start_dt, end_dt, scrip, o=None, sl=0, intra_sl=0, sl_price=None, target_price=None, from_candle_close=False, orderside='SELL', from_next_minute=True, with_ohlc=False, pl_with_slipage=True, per_minute_mtm=False, roundtick=False, pandas=True):
        """
        # Pandas in → Pandas out (default)
        sl_check_single_leg(start_dt, end_dt, scrip)
        
        # Polars in → Polars out
        sl_check_single_leg(start_dt, end_dt, scrip, pandas=False)
        
        # Pandas with MTM
        sl_check_single_leg(start_dt, end_dt, scrip, per_minute_mtm=True)
        
        # Polars MTM (ultra fast)
        sl_check_single_leg(start_dt, end_dt, scrip, per_minute_mtm=True, pandas=False)             
        """
        sl_flag, intra_sl_flag, target_flag, exit_time, pnl = False, False, False, '', 0

        try:
            
            scrip_df:pl.DataFrame = self.get_single_leg_data(start_dt, end_dt, scrip, pandas=False)
            if scrip_df.is_empty(): raise DataEmptyError
            
            o = scrip_df.select("close").item(0) if o is None else o
            slipage = self.Cal_slipage(o) if pl_with_slipage else 0

            if from_next_minute: scrip_df = scrip_df.slice(1)
            if scrip_df.is_empty(): raise DataEmptyError
            
            h = scrip_df.select(pl.col("high").max()).item()
            l = scrip_df.select(pl.col("low").min()).item()
            c = scrip_df.select(pl.col("close").last()).item()
            
            if orderside == 'SELL':
                sl_price_val = (((100 + sl) / 100) * o if sl_price is None else sl_price) if (sl or sl_price) else (h + 1)
                intra_sl_price = ((100 + intra_sl) / 100) * o if intra_sl else (h + 1)
                target_price = target_price if target_price is not None else (l - 1)                
            else:
                sl_price_val = (((100 - sl) / 100) * o if sl_price is None else sl_price) if (sl or sl_price) else (l - 1)
                intra_sl_price = ((100 - intra_sl) / 100) * o if intra_sl else (l - 1)
                target_price = target_price if target_price is not None else (h + 1)
            
            if roundtick or self.market == 'MCX':
                sl_price_val = self.round_to_ticksize(sl_price_val, orderside, 'STOPLOSS')
                intra_sl_price = self.round_to_ticksize(intra_sl_price, orderside, 'STOPLOSS')
                target_price = self.round_to_ticksize(target_price, orderside, 'TARGET')

            if orderside == 'SELL':
                sl_col = "close" if from_candle_close else "high"
                exit_expr = ((pl.col("high") >= intra_sl_price) | (pl.col(sl_col) >= sl_price_val) | (pl.col("low") <= target_price))
            else:
                sl_col = "close" if from_candle_close else "low"
                exit_expr = ((pl.col("low") <= intra_sl_price) | (pl.col(sl_col) <= sl_price_val) | (pl.col("high") >= target_price))
            
            combined_mask = scrip_df.filter(exit_expr).sort("date_time")            
        
            if not combined_mask.is_empty():
                exit_row = combined_mask.row(0, named=True)
                exit_time = exit_row['date_time']

                if orderside == 'SELL':
                    if exit_row['high'] >= intra_sl_price:
                        sl_flag, intra_sl_flag = True, True
                        exit_price = intra_sl_price
                    elif (exit_row['close'] if from_candle_close else exit_row['high']) >= sl_price_val:
                        sl_flag = True
                        exit_price = exit_row['close'] if from_candle_close else sl_price_val 
                    elif exit_row['low'] <= target_price:
                        target_flag = True
                        exit_price = target_price
                elif orderside == 'BUY':
                    if exit_row['low'] <= intra_sl_price:
                        sl_flag, intra_sl_flag = True, True
                        exit_price = intra_sl_price
                    elif (exit_row['close'] if from_candle_close else exit_row['low']) <= sl_price_val:
                        sl_flag = True
                        exit_price = exit_row['close'] if from_candle_close else sl_price_val
                    elif exit_row['high'] >= target_price:
                        target_flag = True
                        exit_price = target_price
            else:
                exit_price = c

            pnl = (exit_price - o) if orderside == 'BUY' else (o - exit_price)
            pnl = round(pnl - slipage, 2)

            # ---------- PER MINUTE MTM ----------
            mtm = None
            if per_minute_mtm:
                scrip_df = scrip_df.to_pandas()
                scrip_df.set_index('date_time', inplace=True)
                if exit_time:
                    scrip_df = scrip_df.loc[scrip_df.index <= exit_time]

                per_minute_mtm_series = o - scrip_df['close'] if orderside == 'SELL' else scrip_df['close'] - o
                per_minute_mtm_series = per_minute_mtm_series - slipage
                per_minute_mtm_series.iloc[-1] = pnl
                if not pandas:
                    per_minute_mtm_series = pl.Series(per_minute_mtm_series)
            
        except DataEmptyError:
            sl_flag, intra_sl_flag, target_flag, exit_time, pnl = False, False, False, '', 0
            o, h, l, c = '', '', '', ''
            per_minute_mtm_series = pd.Series() if pandas else pl.Series()
            sl_price_val = ''
        except Exception as e:
            print('sl_check_single_leg', e)
            traceback.print_exc()
            sl_flag, intra_sl_flag, target_flag, exit_time, pnl = False, False, False, '', 0
            o, h, l, c = '', '', '', ''
            per_minute_mtm_series = pd.Series() if pandas else pl.Series()
            sl_price_val = ''

        sl_price = sl_price_val if (sl or sl_price) else ''

        if with_ohlc:
            ohlc_data = (o, h, l, c, sl_price)
            if per_minute_mtm:
                return (*ohlc_data, exit_time, per_minute_mtm_series)
            
            return (*ohlc_data, sl_flag, intra_sl_flag, target_flag, exit_time, pnl)
        if per_minute_mtm:
            return (exit_time, per_minute_mtm_series)
        return (sl_price, sl_flag, intra_sl_flag, target_flag, exit_time, pnl)

    def _sl_check_combine_leg(self, start_dt, end_dt, ce_scrip, pe_scrip, o=None, sl=0, intra_sl=0, sl_price=None, intra_sl_price=None, target_price=None, orderside='SELL', from_next_minute=True, with_ohlc=False, pl_with_slipage=True, per_minute_mtm=False, roundtick=False, pandas=True):
        """
            # Pandas in → Pandas out (default)
            sl_check_single_leg(start_dt, end_dt, scrip)
            
            # Polars in → Polars out
            sl_check_single_leg(start_dt, end_dt, scrip, pandas=False)
            
            # Pandas with MTM
            sl_check_single_leg(start_dt, end_dt, scrip, per_minute_mtm=True)
            
            # Polars MTM (ultra fast)
            sl_check_single_leg(start_dt, end_dt, scrip, per_minute_mtm=True, pandas=False)             
        """
    
        
        sl_flag, intra_sl_flag, target_flag, exit_time, pnl = False, False, False, '', 0

        try:
            scrip_df = self.get_straddle_data(start_dt, end_dt, ce_scrip, pe_scrip, pandas=False)
            if scrip_df.is_empty(): raise DataEmptyError
            
            o = scrip_df.select("close").item(0) if o is None else o
            slipage = self.Cal_slipage(o) if pl_with_slipage else 0
            
            if from_next_minute: scrip_df = scrip_df.slice(1)
            if scrip_df.is_empty(): raise DataEmptyError
            h, l, cl, ch, c = (scrip_df.select([
                    pl.col("high").max().alias("h"),
                    pl.col("low").min().alias("l"),
                    pl.col("close").min().alias("cl"),
                    pl.col("close").max().alias("ch"),
                    pl.col("close").last().alias("c"),]).row(0))

            if orderside == 'SELL':
                sl_price_val = (((100 + sl) / 100) * o if sl_price is None else sl_price) if (sl or sl_price) else (ch + 1)
                intra_sl_price_val = (((100 + intra_sl) / 100) * o if intra_sl_price is None else intra_sl_price) if (intra_sl or intra_sl_price) else (h + 1)
                target_price = target_price if target_price is not None else (cl - 1)
            
            elif orderside == 'BUY':
                sl_price_val = (((100 - sl) / 100) * o if sl_price is None else sl_price) if (sl or sl_price) else (cl - 1)
                intra_sl_price_val = (((100 - intra_sl) / 100) * o if intra_sl_price is None else intra_sl_price) if (intra_sl or intra_sl_price) else (l - 1)
                target_price = target_price if target_price is not None else (ch + 1)
            
            if roundtick or self.market == 'MCX':
                sl_price_val = self.round_to_ticksize(sl_price_val, orderside, 'STOPLOSS')
                intra_sl_price_val = self.round_to_ticksize(intra_sl_price_val, orderside, 'STOPLOSS')
                target_price = self.round_to_ticksize(target_price, orderside, 'TARGET')

            if orderside == 'SELL':
                exit_expr = ((pl.col("high") >= intra_sl_price_val) | (pl.col("close") >= sl_price_val) | (pl.col("close") <= target_price))
            else:
                exit_expr = ((pl.col("low") <= intra_sl_price_val) | (pl.col("close") <= sl_price_val) | (pl.col("close") >= target_price))
            
            combined_mask = scrip_df.filter(exit_expr).sort("date_time")
            if not combined_mask.is_empty():
                exit_row = combined_mask.row(0, named=True)
                exit_time = exit_row['date_time']
                if orderside == 'SELL':
                    if exit_row['high'] >= intra_sl_price_val:
                        sl_flag, intra_sl_flag = True, True
                        exit_price = intra_sl_price_val
                    elif exit_row['close'] >= sl_price_val:
                        sl_flag = True
                        exit_price = exit_row['close']
                    elif exit_row['close'] <= target_price:
                        target_flag = True
                        exit_price = exit_row['close']
                elif orderside == 'BUY':
                    if exit_row['low'] <= intra_sl_price_val:
                        sl_flag, intra_sl_flag = True, True
                        exit_price = intra_sl_price_val
                    elif exit_row['close'] <= sl_price_val:
                        sl_flag = True
                        exit_price = exit_row['close']
                    elif exit_row['close'] >= target_price:
                        target_flag = True
                        exit_price = exit_row['close']
            else:
                exit_price = c
            pnl = (exit_price - o) if orderside == 'BUY' else (o - exit_price)
            pnl = round(pnl - slipage, 2)
            
            if per_minute_mtm:
                scrip_df = scrip_df.to_pandas()
                scrip_df.set_index('date_time', inplace=True)
                if exit_time:
                    scrip_df = scrip_df.loc[scrip_df.index <= exit_time]
                per_minute_mtm_series = o - scrip_df['close'] if orderside == 'SELL' else scrip_df['close'] - o
                per_minute_mtm_series = per_minute_mtm_series - slipage
                per_minute_mtm_series.iloc[-1] = pnl
                if not pandas:
                    per_minute_mtm_series = pl.Series(per_minute_mtm_series)
            
        except DataEmptyError:
            sl_flag, intra_sl_flag, target_flag, exit_time, pnl = False, False, False, '', 0
            o, h, l, c = '', '', '', ''
            per_minute_mtm_series = pd.Series() if pandas else pl.Series()
            sl_price_val, intra_sl_price_val = '', ''
        except Exception as e:
            print('sl_check_combine_leg', e)
            traceback.print_exc()
            sl_flag, intra_sl_flag, target_flag, exit_time, pnl = False, False, False, '', 0
            o, h, l, c = '', '', '', ''
            per_minute_mtm_series = pd.Series() if pandas else pl.Series()
            sl_price_val, intra_sl_price_val = '', ''
        sl_price = sl_price_val if (sl or sl_price) else ''
        intra_sl_price = intra_sl_price_val if (intra_sl or intra_sl_price) else ''
        
        if with_ohlc:
            ohlc_data = (o, h, l, c, sl_price, intra_sl_price)
            if per_minute_mtm:
                return (*ohlc_data, exit_time, per_minute_mtm_series)
            return (*ohlc_data, sl_flag, intra_sl_flag, target_flag, exit_time, pnl)
        if per_minute_mtm:
            return (exit_time, per_minute_mtm_series)
        return (sl_price, intra_sl_price, sl_flag, intra_sl_flag, target_flag, exit_time, pnl)
            
    def decay_check_by_given_data(self, scrip_df, decay=None, decay_price=None, from_candle_close=False, orderside='SELL', from_next_minute=True, with_ohlc=False, roundtick=False, pandas=True):
        """
            # Pandas in → Pandas out (default)
            decay_check_by_given_data(start_dt, end_dt, scrip)
            
            # Polars in → Polars out
            decay_check_by_given_data(start_dt, end_dt, scrip, pandas=False)
            
            # Pandas with MTM
            decay_check_by_given_data(start_dt, end_dt, scrip, per_minute_mtm=True)
            
            # Polars MTM (ultra fast)
            decay_check_by_given_data(start_dt, end_dt, scrip, per_minute_mtm=True, pandas=False)             
        """
        decay_flag, decay_time = False, ''
        try:
            if pandas:
                if scrip_df.empty: raise DataEmptyError
                scrip_df:pl.DataFrame = pl.from_pandas(scrip_df.reset_index())
            else:
                if scrip_df.is_empty(): raise DataEmptyError
            start_dt = scrip_df.select("date_time").item(0)
            o = scrip_df.select("close").item(0)
            
            if from_next_minute: scrip_df = scrip_df.slice(1)
            if scrip_df.is_empty(): raise DataEmptyError
            
            if with_ohlc:
                stats = scrip_df.select([
                pl.col("high").max().alias("h"),
                pl.col("low").min().alias("l"),
                pl.col("close").last().alias("c"),
                ]).row(0)
                h, l, c = stats
            if decay == 0 or decay_price == -1:
                decay_price = o
                decay_flag = True
                decay_time = start_dt
            else:
                if orderside == 'SELL':
                    decay_price = ((100 - decay)/100) * o if decay_price is None else decay_price
                    
                    if roundtick or self.market == 'MCX':
                        decay_price = self.round_to_ticksize(decay_price, orderside, 'DECAY')

                    decay_col = "close" if from_candle_close else "low"
                    decay_expr = pl.col(decay_col) <= decay_price
                    
                elif orderside == 'BUY':
                    decay_price = ((100 + decay)/100) * o if decay_price is None else decay_price
                    
                    if roundtick or self.market == 'MCX':
                        decay_price = self.round_to_ticksize(decay_price, orderside, 'DECAY')

                    decay_col = "close" if from_candle_close else "high"
                    decay_expr = pl.col(decay_col) >= decay_price

                mask_decay = scrip_df.filter(decay_expr)
                if not mask_decay.is_empty():
                    decay_flag = True
                    decay_time = mask_decay.select("date_time").item(0)
            
        except DataEmptyError:
            decay_flag, decay_time = False, ''
            o, h, l, c = '', '', '', ''
        except Exception as e:
            print('decay_check_single_leg', e)
            traceback.print_exc()
            decay_flag, decay_time = False, ''
            o, h, l, c = '', '', '', ''
        
        if with_ohlc:
            return o, h, l, c, decay_price, decay_flag, decay_time
        return decay_price, decay_flag, decay_time
           
    def _decay_check_single_leg(self, start_dt, end_dt, scrip, decay=None, decay_price=None, from_candle_close=False, orderside='SELL', from_next_minute=True, with_ohlc=False, roundtick=False):
        """
            # Pandas in → Pandas out (default)
            decay_check_single_leg(start_dt, end_dt, scrip)
            
            # Polars in → Polars out
            decay_check_single_leg(start_dt, end_dt, scrip, pandas=False)
            
            # Pandas with MTM
            decay_check_single_leg(start_dt, end_dt, scrip, per_minute_mtm=True)
            
            # Polars MTM (ultra fast)
            decay_check_single_leg(start_dt, end_dt, scrip, per_minute_mtm=True, pandas=False)             
        """
        decay_flag, decay_time = False, ''
        o = h = l = c = ''

        try:
            scrip_df:pl.DataFrame = self.get_single_leg_data(start_dt, end_dt, scrip, pandas=False)
            if scrip_df.is_empty(): raise DataEmptyError
            
            o = scrip_df.select("close").item(0)

            if from_next_minute: scrip_df = scrip_df.slice(1)
            if scrip_df.is_empty(): raise DataEmptyError
            
            if with_ohlc:
                stats = scrip_df.select([
                pl.col("high").max().alias("h"),
                pl.col("low").min().alias("l"),
                pl.col("close").last().alias("c"),
                ]).row(0)
                h, l, c = stats
            if decay == 0 or decay_price == -1:
                decay_price = o
                decay_flag = True
                decay_time = start_dt
            else:
                if orderside == 'SELL':
                    decay_price = ((100 - decay)/100) * o if decay_price is None else decay_price
                    if roundtick or self.market == 'MCX':
                        decay_price = self.round_to_ticksize(decay_price, orderside, 'DECAY')
                    decay_col = "close" if from_candle_close else "low"
                    decay_expr = pl.col(decay_col) <= decay_price
                elif orderside == 'BUY':
                    decay_price = ((100 + decay)/100) * o if decay_price is None else decay_price
                    if roundtick or self.market == 'MCX':
                        decay_price = self.round_to_ticksize(decay_price, orderside, 'DECAY')
                    decay_col = "close" if from_candle_close else "high"
                    decay_expr = pl.col(decay_col) >= decay_price
                mask_decay = scrip_df.filter(decay_expr)
                if not mask_decay.is_empty():
                    decay_flag = True
                    decay_time = mask_decay.select("date_time").item(0)
            
            
        except DataEmptyError:
            decay_flag, decay_time = False, ''
            o, h, l, c = '', '', '', ''
        except Exception as e:
            print('decay_check_single_leg', e)
            traceback.print_exc()
            decay_flag, decay_time = False, ''
            o, h, l, c = '', '', '', ''
        if with_ohlc:
            return o, h, l, c, decay_price, decay_flag, decay_time
        return decay_price, decay_flag, decay_time
        
    def _sl_check_single_leg_with_sl_trail(self, start_dt, end_dt, scrip, o=None, sl=0, sl_price=None, trail=0, sl_trail=0, from_candle_close=False, orderside='SELL', from_next_minute=True, with_ohlc=False, pl_with_slipage=True, per_minute_mtm=False, roundtick=False, pandas=True):
        sl_flag, trail_flag, exit_time, pnl = False, False, '', 0
        h, l, c = None, None, None
        try:
            scrip_df:pl.DataFrame = self.get_single_leg_data(start_dt, end_dt, scrip, pandas=False)
            if scrip_df.is_empty(): raise DataEmptyError
            o = scrip_df.select("close").item(0) if o is None else o
            slipage = self.Cal_slipage(o) if pl_with_slipage else 0
            
            if from_next_minute: scrip_df = scrip_df.slice(1)
            if scrip_df.is_empty(): raise DataEmptyError
            
            if with_ohlc:
                stats = scrip_df.select([
                pl.col("high").max().alias("h"),
                pl.col("low").min().alias("l"),
                ]).row(0)
                h, l= stats
            c = scrip_df.select(pl.col("close").last()).item()
            
            exit_price = None
            if orderside == 'SELL':
                sl_price = ((100 + sl) / 100) * o if sl_price is None else sl_price
                
                if roundtick or self.market == 'MCX':
                    sl_price = self.round_to_ticksize(sl_price, orderside, 'STOPLOSS')

                if (trail != 0) and (sl_trail != 0):
                    
                    trail_price = ((100 - trail) / 100) * o

                    if roundtick or self.market == 'MCX':
                        trail_price = self.round_to_ticksize(trail_price, orderside, 'TARGET')
                        
                    for row in scrip_df.iter_rows(named=True):
                        
                        if (from_candle_close and row['close'] >= sl_price) or (not from_candle_close and row['high'] >= sl_price):
                            sl_flag = True
                            exit_time = row['date_time']
                            exit_price = row['close'] if from_candle_close else sl_price
                            break
                        elif (from_candle_close and row['close'] <= trail_price) or (not from_candle_close and row['low'] <= trail_price):
                            trail_flag = True
                                
                            sl_price = sl_price * (1 - (sl_trail/100))
                            trail_price = trail_price * (1 - (trail/100))
                            
                            if roundtick or self.market == 'MCX':
                                sl_price = self.round_to_ticksize(sl_price, orderside, 'STOPLOSS')
                                trail_price = self.round_to_ticksize(trail_price, orderside, 'TARGET')
                            
                elif (trail == 0) and (sl_trail != 0) and (sl == sl_trail):
                    # trailing at every minute
                    
                    for row in scrip_df.iter_rows(named=True):
                        
                        if (from_candle_close and row['close'] >= sl_price) or (not from_candle_close and row['high'] >= sl_price):
                            sl_flag = True
                            exit_time = row['date_time']
                            exit_price = row['close'] if from_candle_close else sl_price
                            break
                        else:
                            sl_price = min(sl_price, ((100 + sl) / 100) * row['close']) if from_candle_close else min(sl_price, ((100 + sl) / 100) * row['low'])
                            if roundtick or self.market == 'MCX':
                                sl_price = self.round_to_ticksize(sl_price, orderside, 'STOPLOSS')
            
            elif orderside == 'BUY':
                sl_price = ((100 - sl) / 100) * o if sl_price is None else sl_price
                if roundtick or self.market == 'MCX':
                    sl_price = self.round_to_ticksize(sl_price, orderside, 'STOPLOSS')
                
                if (trail != 0) and (sl_trail != 0):
                    
                    trail_price = ((100 + trail) / 100) * o

                    if roundtick or self.market == 'MCX':
                        trail_price = self.round_to_ticksize(trail_price, orderside, 'TARGET')

                    for row in scrip_df.iter_rows(named=True):
                        
                        if (from_candle_close and row['close'] <= sl_price) or (not from_candle_close and row['low'] <= sl_price):
                            sl_flag = True
                            exit_time = row['date_time']
                            exit_price = row['close'] if from_candle_close else sl_price
                            break
                        elif (from_candle_close and row['close'] >= trail_price) or (not from_candle_close and row['high'] >= trail_price):
                            trail_flag = True
                                
                            sl_price = sl_price * (1 + (sl_trail/100))
                            trail_price = trail_price * (1 + (trail/100))
                            
                            if roundtick or self.market == 'MCX':
                                sl_price = self.round_to_ticksize(sl_price, orderside, 'STOPLOSS')
                                trail_price = self.round_to_ticksize(trail_price, orderside, 'TARGET')
                
                elif (trail == 0) and (sl_trail != 0) and (sl == sl_trail):
                    # trailing at every minute
                    
                    for row in scrip_df.iter_rows(named=True):
                        
                        if (from_candle_close and row['close'] <= sl_price) or (not from_candle_close and row['low'] <= sl_price):
                            sl_flag = True
                            exit_time = row['date_time']
                            exit_price = row['close'] if from_candle_close else sl_price
                            break
                        else:
                            sl_price = max(sl_price, ((100 - sl) / 100) * row['close']) if from_candle_close else max(sl_price, ((100 - sl) / 100) * row['high'])
                            if roundtick or self.market == 'MCX':
                                sl_price = self.round_to_ticksize(sl_price, orderside, 'STOPLOSS')
            
            exit_price = exit_price if exit_price is not None else c
            
            pnl = (exit_price - o) if orderside == 'BUY' else (o - exit_price)
            pnl = round(pnl - slipage, 2)
            
            if per_minute_mtm:
                scrip_df = scrip_df.to_pandas()
                scrip_df.set_index('date_time', inplace=True)
                if exit_time:
                    scrip_df = scrip_df.loc[scrip_df.index <= exit_time]

                per_minute_mtm_series = o - scrip_df['close'] if orderside == 'SELL' else scrip_df['close'] - o
                per_minute_mtm_series = per_minute_mtm_series - slipage
                per_minute_mtm_series.iloc[-1] = pnl
                if not pandas:
                    per_minute_mtm_series = pl.Series(per_minute_mtm_series)
            
            
        except DataEmptyError:
            sl_flag, trail_flag, exit_time, pnl = False, False, '', 0
            o, h, l, c = '', '', '', ''
            per_minute_mtm_series = pd.Series() if pandas else pl.Series()
            sl_price = ''
        except Exception as e:
            print('sl_check_single_leg_with_sl_trail', e)
            traceback.print_exc()
            sl_flag, trail_flag, exit_time, pnl = False, False, '', 0
            o, h, l, c = '', '', '', ''
            per_minute_mtm_series = pd.Series() if pandas else pl.Series()
            sl_price = ''
        
        if with_ohlc:
            ohlc_data = (o, h, l, c, sl_price)
            if per_minute_mtm:
                return (*ohlc_data, exit_time, per_minute_mtm_series)
            return (*ohlc_data, sl_flag, trail_flag, exit_time, pnl)
        if per_minute_mtm:
            return (exit_time, per_minute_mtm_series)
        return (sl_price, sl_flag, trail_flag, exit_time, pnl)
        
    def _sl_check_combine_leg_with_sl_trail(self, start_dt, end_dt, ce_scrip, pe_scrip, o=None, sl=0, intra_sl=0, sl_price=None, intra_sl_price=None, trail=0, sl_trail=0, orderside='SELL', from_next_minute=True, with_ohlc=False, pl_with_slipage=True, per_minute_mtm=False, roundtick=False, pandas=True):
        sl_flag, intra_sl_flag, trail_flag, exit_time, pnl = False, False, False, '', 0

        try:
            scrip_df:pl.DataFrame = self.get_straddle_data(start_dt, end_dt, ce_scrip, pe_scrip, pandas = False)
            if scrip_df.is_empty(): raise DataEmptyError

            o = scrip_df.select(pl.col("close").first()).item() if o is None else o
            slipage = self.Cal_slipage(o) if pl_with_slipage else 0

            if from_next_minute: scrip_df = scrip_df.slice(1)
            if scrip_df.is_empty(): raise DataEmptyError
            
            stats = scrip_df.select([
                pl.col("high").max().alias("h"),
                pl.col("low").min().alias("l"),
                pl.col("close").min().alias("cl"),
                pl.col("close").max().alias("ch"),
                pl.col("close").last().alias("c"),
            ]).row(0)
            h, l, cl, ch, c = stats
            exit_price = None
            if orderside == 'SELL':
                sl_price_val = (((100 + sl) / 100) * o if sl_price is None else sl_price) if (sl or sl_price) else (ch + 1)
                intra_sl_price_val = (((100 + intra_sl) / 100) * o if intra_sl_price is None else intra_sl_price) if (intra_sl or intra_sl_price) else (h + 1)

                if roundtick or self.market == 'MCX':
                    sl_price_val = self.round_to_ticksize(sl_price_val, orderside, 'STOPLOSS')
                    intra_sl_price_val = self.round_to_ticksize(intra_sl_price_val, orderside, 'STOPLOSS')
                
                if (trail != 0) and (sl_trail != 0):
                    
                    trail_price = ((100 - trail) / 100) * o
                    
                    if roundtick or self.market == 'MCX':
                        trail_price = self.round_to_ticksize(trail_price, orderside, 'TARGET')
                    
                    for row in scrip_df.select(["date_time", "open", "high", "low", "close"]).iter_rows(named=True):

                        if ((sl or sl_price) and row["close"] >= sl_price_val) or ((intra_sl or intra_sl_price) and row["high"] >= intra_sl_price_val):
                            sl_flag = True
                            intra_sl_flag = True if ((intra_sl or intra_sl_price) and row["high"] >= intra_sl_price_val) else False
                            exit_time = row["date_time"]
                            exit_price = intra_sl_price_val if ((intra_sl or intra_sl_price) and row["high"] >= intra_sl_price_val) else row["close"]
                            break

                        elif ((sl or sl_price) and row["close"] <= trail_price) or ((intra_sl or intra_sl_price) and row["low"] <= trail_price):
                            trail_flag = True
                            
                            sl_price_val = sl_price_val * (1 - (sl_trail/100))
                            intra_sl_price_val = intra_sl_price_val * (1 - (sl_trail/100))
                            trail_price = trail_price * (1 - (trail/100))
                            
                            if roundtick or self.market == 'MCX':
                                sl_price_val = self.round_to_ticksize(sl_price_val, orderside, 'STOPLOSS')
                                intra_sl_price_val = self.round_to_ticksize(intra_sl_price_val, orderside, 'STOPLOSS')
                                trail_price = self.round_to_ticksize(trail_price, orderside, 'TARGET')
                            
                elif (trail == 0) and (sl_trail != 0) and ((sl == sl_trail) or (intra_sl == sl_trail)):
                    # trailing at every minute
                    
                    for row in scrip_df.select(["date_time", "open", "high", "low", "close"]).iter_rows(named=True):

                        if ((sl or sl_price) and row["close"] >= sl_price_val) or ((intra_sl or intra_sl_price) and row["high"] >= intra_sl_price_val):
                            sl_flag = True
                            intra_sl_flag = True if ((intra_sl or intra_sl_price) and row["high"] >= intra_sl_price_val) else False
                            exit_time = row["date_time"]
                            exit_price = intra_sl_price_val if ((intra_sl or intra_sl_price) and row["high"] >= intra_sl_price_val) else row["close"]
                            break
                        else:
                            sl_price_val = min(sl_price_val, ((100 + sl) / 100) * row["close"]) if sl else (ch + 1)
                            intra_sl_price_val = min(intra_sl_price_val, ((100 + intra_sl) / 100) * row["low"]) if intra_sl else (h + 1)

                            if roundtick or self.market == 'MCX':
                                sl_price_val = self.round_to_ticksize(sl_price_val, orderside, 'STOPLOSS')
                                intra_sl_price_val = self.round_to_ticksize(intra_sl_price_val, orderside, 'STOPLOSS')                    
                    
            elif orderside == 'BUY':
                
                sl_price_val = (((100 - sl) / 100) * o if sl_price is None else sl_price) if (sl or sl_price) else (cl - 1)
                intra_sl_price_val = (((100 - intra_sl) / 100) * o if intra_sl_price is None else intra_sl_price) if (intra_sl or intra_sl_price) else (l - 1)

                if roundtick or self.market == 'MCX':
                    sl_price_val = self.round_to_ticksize(sl_price_val, orderside, 'STOPLOSS')
                    intra_sl_price_val = self.round_to_ticksize(intra_sl_price_val, orderside, 'STOPLOSS')
                
                if (trail != 0) and (sl_trail != 0):
                    
                    trail_price = ((100 + trail) / 100) * o
                    
                    if roundtick or self.market == 'MCX':
                        trail_price = self.round_to_ticksize(trail_price, orderside, 'TARGET')
                    
                    for row in scrip_df.select(["date_time", "open", "high", "low", "close"]).iter_rows(named=True):

                        if ((sl or sl_price) and row["close"] <= sl_price_val) or ((intra_sl or intra_sl_price) and row["low"] <= intra_sl_price_val):
                            sl_flag = True
                            intra_sl_flag = True if ((intra_sl or intra_sl_price) and row["low"] <= intra_sl_price_val) else False
                            exit_time = row["date_time"]
                            exit_price = intra_sl_price_val if ((intra_sl or intra_sl_price) and row["low"] <= intra_sl_price_val) else row["close"]
                            break

                        elif ((sl or sl_price) and row["close"] >= trail_price) or ((intra_sl or intra_sl_price) and row["high"] >= trail_price):
                            trail_flag = True
                            sl_price_val = sl_price_val * (1 + (sl_trail/100))
                            intra_sl_price_val = intra_sl_price_val * (1 + (sl_trail/100))
                            trail_price = trail_price * (1 + (trail/100))
                            
                            if roundtick or self.market == 'MCX':
                                sl_price_val = self.round_to_ticksize(sl_price_val, orderside, 'STOPLOSS')
                                intra_sl_price_val = self.round_to_ticksize(intra_sl_price_val, orderside, 'STOPLOSS')
                                trail_price = self.round_to_ticksize(trail_price, orderside, 'TARGET')
                            
                elif (trail == 0) and (sl_trail != 0) and ((sl == sl_trail) or (intra_sl == sl_trail)):
                    # trailing at every minute
                    
                    for row in scrip_df.select(["date_time", "open", "high", "low", "close"]).iter_rows(named=True):

                        if ((sl or sl_price) and row["close"] <= sl_price_val) or ((intra_sl or intra_sl_price) and row["low"] <= intra_sl_price_val):
                            sl_flag = True
                            intra_sl_flag = True if ((intra_sl or intra_sl_price) and row["low"] <= intra_sl_price_val) else False
                            exit_time = row["date_time"]
                            exit_price = intra_sl_price_val if ((intra_sl or intra_sl_price) and row["low"] <= intra_sl_price_val) else row["close"]
                            break
                        else:
                            sl_price_val = max(sl_price_val, ((100 - sl) / 100) * row["close"]) if sl else (cl - 1)
                            intra_sl_price_val = max(intra_sl_price_val, ((100 - intra_sl) / 100) * row["high"]) if intra_sl else (l - 1)

                            if roundtick or self.market == 'MCX':
                                sl_price_val = self.round_to_ticksize(sl_price_val, orderside, 'STOPLOSS')
                                intra_sl_price_val = self.round_to_ticksize(intra_sl_price_val, orderside, 'STOPLOSS')

            exit_price = c if exit_price is None else exit_price

            pnl = (exit_price - o) if orderside == 'BUY' else (o - exit_price)
            pnl = round(pnl - slipage, 2)

            if per_minute_mtm:
                scrip_df = scrip_df.to_pandas()
                scrip_df.set_index('date_time', inplace=True)
                if exit_time:
                    scrip_df.index = pd.to_datetime(scrip_df.index)
                    exit_time = pd.to_datetime(exit_time)
                    scrip_df = scrip_df.loc[scrip_df.index <= exit_time]

                per_minute_mtm_series = o - scrip_df['close'] if orderside == 'SELL' else scrip_df['close'] - o
                per_minute_mtm_series = per_minute_mtm_series - slipage
                per_minute_mtm_series.iloc[-1] = pnl
                if not pandas:
                    per_minute_mtm_series = pl.Series(per_minute_mtm_series)

        except DataEmptyError:
            sl_flag, intra_sl_flag, trail_flag, exit_time, pnl = False, False, False, '', 0
            o, h, l, c = '', '', '', ''
            per_minute_mtm_series = pd.Series() if pandas else pl.Series()
            sl_price_val, intra_sl_price_val = '', ''
        except Exception as e:
            print('sl_check_combine_leg_with_sl_trail', e)
            traceback.print_exc()
            sl_flag, intra_sl_flag, trail_flag, exit_time, pnl = False, False, False, '', 0
            o, h, l, c = '', '', '', ''
            per_minute_mtm_series = pd.Series() if pandas else pl.Series()
            sl_price_val, intra_sl_price_val = '', ''

        sl_price = sl_price_val if (sl or sl_price) else ''
        intra_sl_price = intra_sl_price_val if (intra_sl or intra_sl_price) else ''

        if with_ohlc:
            ohlc_data = (o, h, l, c, sl_price, intra_sl_price)
            if per_minute_mtm:
                return (*ohlc_data, exit_time, per_minute_mtm_series)
            return (*ohlc_data, sl_flag, intra_sl_flag, trail_flag, exit_time, pnl)
        if per_minute_mtm:
            return (exit_time, per_minute_mtm_series)
        return (sl_price, intra_sl_price, sl_flag, intra_sl_flag, trail_flag, exit_time, pnl)

    def _straddle_indicator(self, start_dt, end_dt, si_indicator, si_buffer, buffer_min):

        buffer_start = max(datetime.datetime.combine(start_dt.date(), self.meta_start_time), start_dt - datetime.timedelta(minutes=buffer_min))
        buffer_range = pd.date_range(buffer_start, start_dt - datetime.timedelta(minutes=1), freq='1min')
        
        std_prices = [self.get_strike(dt, dt+datetime.timedelta(minutes=1))[2:4] for dt in buffer_range]
        valid_std_prices = [(ce + pe) for ce, pe in std_prices if ce is not None and pe is not None]
        
        if not valid_std_prices:
            return False, ''
        
        if si_indicator == 'LOW':
            extreme = np.min(valid_std_prices)
        elif si_indicator == 'HIGH':
            extreme = np.max(valid_std_prices)
        elif si_indicator == 'AVG':
            extreme = np.mean(valid_std_prices)

        threshold = float(extreme) * (1 + si_buffer)
        for dt in pd.date_range(start_dt, end_dt - datetime.timedelta(minutes=5), freq='1min'):
            
            _, _, ce_price, pe_price, _, entry_time = self.get_strike(dt, dt+datetime.timedelta(minutes=1))
            if entry_time is not None:
                if (ce_price + pe_price) <= threshold:
                    return True, entry_time

        return False, ''
    
    def __del__(self) -> None:
        print("Deleting instance ...", self.current_date)


class WeeklyBacktest(IntradayBacktest):

    def __init__(self, pickle_path, index, week_dates, from_dte, to_dte, start_time, end_time):
        
        self.pickle_path, self.index, self.week_dates, self.from_dte, self.to_dte, self.meta_start_time, self.meta_end_time = pickle_path, index, week_dates, from_dte, to_dte, start_time, end_time
        
        self.current_week_dates = sorted(set(([self.week_dates[0]] * (99 - len(self.week_dates)) + self.week_dates)[-from_dte : None if to_dte == 1 else -to_dte + 1]))
        self.__future_pickle_path, self.__option_pickle_path = self.get_future_option_path(index)
        
        future_data_list = []
        for current_date in self.current_week_dates:
            future_parquet_path = self.__future_pickle_path.format(date=current_date.date(), extn='parquet')
            future_pickle_path = self.__future_pickle_path.format(date=current_date.date(), extn='pkl')

            if os.path.exists(future_parquet_path):
                future_data_list.append(pl.read_parquet(future_parquet_path))
            elif os.path.exists(future_pickle_path):
                future_data_list.append(pl.from_pandas(pd.read_pickle(future_pickle_path)))

        if not future_data_list:
            raise FileNotFoundError(f"No future data files found for {self.index} for the given week.")
        
        self.future_data_pl = pl.concat(future_data_list).sort("date_time").select(["date_time", "open", "high", "low", "close"])
        self.future_data_pl = self.future_data_pl.with_columns(pl.col("date_time").cast(pl.Datetime))
        option_data_list = []
        for current_date in self.current_week_dates:
            option_parquet_path = self.__option_pickle_path.format(date=current_date.date(), extn='parquet')
            option_pickle_path = self.__option_pickle_path.format(date=current_date.date(), extn='pkl')

            if os.path.exists(option_parquet_path):
                option_data_list.append(pl.read_parquet(option_parquet_path))
            elif os.path.exists(option_pickle_path):
                option_data_list.append(pl.from_pandas(pd.read_pickle(option_pickle_path)))
        
        self.options_pl = pl.concat(option_data_list).sort("date_time").select(["scrip", "date_time", "open", "high", "low", "close"])
        self.options_pl = self.options_pl.with_columns(pl.col("date_time").cast(pl.Datetime), pl.col("scrip").cast(pl.Utf8))
        self.options_pl = self.options_pl.filter((pl.col("date_time").dt.time() >= start_time) & (pl.col("date_time").dt.time() <= end_time))
        
        self.gap = self.get_gap()
        self.tick_size = self.TICKS.get(index.lower(), 0.05)
        
        if self.index in NSE_INDICES:
            self.market = 'NSE'
        elif self.index in BSE_INDICES:
            self.market = 'BSE'
        elif self.index in MCX_INDICES:
            self.market = 'MCX'
        elif self.index in US_INDICES:
            self.market = 'US'
        else:
            self.market = 'OTHER'

        self.get_single_leg_data = lru_cache(maxsize=4096)(self._get_single_leg_data)
        self.get_straddle_data = lru_cache(maxsize=4096)(self._get_straddle_data)
        self.get_strike = lru_cache(maxsize=4096)(self._get_strike)
        self.sl_check_single_leg = lru_cache(maxsize=4096)(self._sl_check_single_leg)
        self.sl_check_combine_leg = lru_cache(maxsize=4096)(self._sl_check_combine_leg)
        self.decay_check_single_leg = lru_cache(maxsize=4096)(self._decay_check_single_leg)
        self.sl_check_single_leg_with_sl_trail = lru_cache(maxsize=4096)(self._sl_check_single_leg_with_sl_trail)
        self.sl_check_combine_leg_with_sl_trail = lru_cache(maxsize=4096)(self._sl_check_combine_leg_with_sl_trail)
        self.straddle_indicator = lru_cache(maxsize=4096)(self._straddle_indicator)

        self.get_EOD_straddle_strike = lru_cache(maxsize=4096)(self._get_EOD_straddle_strike)
        self.sl_range_check_combine_leg = lru_cache(maxsize=4096)(self._sl_range_check_combine_leg)
        self.sl_range_trail_check_combine_leg = lru_cache(maxsize=4096)(self._sl_range_trail_check_combine_leg)

    def get_synthetic_future(self, straddle_strike, ce_price, pe_price):
        synthetic_future = straddle_strike + ce_price - pe_price
        return synthetic_future
        
    def get_sl_range(self, strike, premium, range_sl, intra_range_sl):
        range_limit = premium * (range_sl/100)
        lower_range = strike - range_limit
        upper_range = strike + range_limit
        
        if intra_range_sl:
            intra_range_limit = premium * (intra_range_sl/100)
            intra_lower_range = strike - intra_range_limit
            intra_upper_range = strike + intra_range_limit
            return lower_range, upper_range, intra_lower_range, intra_upper_range
        else:
            return lower_range, upper_range
        
    def get_straddle_strike(self, start_dt, end_dt, sd=0, SDroundoff=False):

        future_pl = ( self.future_data_pl.filter( (pl.col("date_time") >= start_dt) & (pl.col("date_time") <= end_dt) ) .select(["date_time", "close"]) .sort("date_time") )
        current_date = start_dt.date()
        for row in future_pl.iter_rows(named=True):
            current_dt = row["date_time"]
            if current_dt.date() != current_date:
                break
            try:
                future_price = row["close"]
                # print(self.gap)
                round_future_price = round(future_price/self.gap)*self.gap
                ce_scrip, pe_scrip = f"{round_future_price}CE", f"{round_future_price}PE" 
                
                ce_price = self._get_option_close(current_dt, ce_scrip) 
                pe_price = self._get_option_close(current_dt, pe_scrip)

                if ce_price is None or pe_price is None: continue 
                
                # Synthetic future 
                syn_future = ce_price - pe_price + round_future_price 
                round_syn_future = round(syn_future/self.gap)*self.gap

                # Scrip lists 
                ce_scrip_list = [f"{round_syn_future}CE", f"{round_syn_future+self.gap}CE", f"{round_syn_future-self.gap}CE"] 
                pe_scrip_list = [f"{round_syn_future}PE", f"{round_syn_future+self.gap}PE", f"{round_syn_future-self.gap}PE"]
                
                selected, min_value = None, float("inf")
                for i in range(3): 
                    try: 
                        ce_price = self._get_option_close(current_dt, ce_scrip_list[i]) 
                        pe_price = self._get_option_close(current_dt, pe_scrip_list[i]) 
                        if ce_price is None or pe_price is None: continue 
                        diff = abs(ce_price-pe_price) 
                        if min_value > diff: 
                            min_value = diff 
                            selected = (ce_scrip_list[i], pe_scrip_list[i], ce_price, pe_price) 
                    except: pass
                if selected is None: continue
                ce_scrip, pe_scrip, ce_price, pe_price = selected

                if sd: 
                    sd_range = (ce_price + pe_price) * sd 
                    
                    if SDroundoff: 
                        sd_range = round(sd_range/self.gap)*self.gap 
                    else: 
                        sd_range = max(self.gap, round(sd_range/self.gap)*self.gap)
                
                    ce_scrip, pe_scrip = f"{get_strike(ce_scrip) + sd_range}CE", f"{get_strike(pe_scrip) - sd_range}PE" 
                    ce_price, pe_price = self._get_option_close(current_dt, ce_scrip), self._get_option_close(current_dt, pe_scrip)

                    if ce_price is None or pe_price is None: continue
                
                return ce_scrip, pe_scrip, ce_price, pe_price, future_price, current_dt     
            except (IndexError, KeyError, ValueError, TypeError): 
                if current_dt.date() != current_date: break
            except Exception as e: 
                print('get_straddle_strike', e) 
                # traceback.print_exc()
                if current_dt.date() != current_date: break
        return None, None, None, None, None, None 
    
    def _get_EOD_straddle_strike(self, current_date):
        
        check_limit = 15 #min
        start_dt = datetime.datetime.combine(current_date, self.meta_end_time)
        end_dt = start_dt - datetime.timedelta(minutes=check_limit)
        
        # Create lookup dictionaries from Polars for faster access
        future_dict = {row['date_time']: row for row in self.future_data_pl.iter_rows(named=True)}
        options_dict = {(row['date_time'], row['scrip']): row for row in self.options_pl.iter_rows(named=True)}
        
        while start_dt > end_dt:
            try:
                # find strike nearest to future price
                future_row = future_dict.get(start_dt)
                if future_row is None:
                    start_dt -= datetime.timedelta(minutes=1)
                    continue
                future_price = future_row['close']
                round_future_price = round(future_price/self.gap)*self.gap

                ce_scrip, pe_scrip = f"{round_future_price}CE", f"{round_future_price}PE"
                ce_row = options_dict.get((start_dt, ce_scrip))
                pe_row = options_dict.get((start_dt, pe_scrip))
                if ce_row is None or pe_row is None:
                    start_dt -= datetime.timedelta(minutes=1)
                    continue
                ce_price, pe_price = ce_row['close'], pe_row['close']
                
                # Synthetic future
                syn_future = ce_price - pe_price + round_future_price
                round_syn_future = round(syn_future/self.gap)*self.gap

                # Scrip lists
                ce_scrip_list = [f"{round_syn_future}CE", f"{round_syn_future+self.gap}CE", f"{round_syn_future-self.gap}CE"]
                pe_scrip_list = [f"{round_syn_future}PE", f"{round_syn_future+self.gap}PE", f"{round_syn_future-self.gap}PE"]
                
                scrip_index, min_value = None, float("inf")
                for i in range(3):
                    try:
                        ce_row_i = options_dict.get((start_dt, ce_scrip_list[i]))
                        pe_row_i = options_dict.get((start_dt, pe_scrip_list[i]))
                        if ce_row_i is None or pe_row_i is None:
                            continue
                        ce_price = ce_row_i['close']
                        pe_price = pe_row_i['close']
                        diff = abs(ce_price-pe_price)
                        if min_value > diff:
                            min_value = diff
                            scrip_index = i
                    except:
                        pass
                        
                # Required scrip and their price
                if scrip_index is None:
                    start_dt -= datetime.timedelta(minutes=1)
                    continue
                ce_scrip, pe_scrip = ce_scrip_list[scrip_index], pe_scrip_list[scrip_index]
                ce_row_final = options_dict.get((start_dt, ce_scrip))
                pe_row_final = options_dict.get((start_dt, pe_scrip))
                if ce_row_final is None or pe_row_final is None:
                    start_dt -= datetime.timedelta(minutes=1)
                    continue
                ce_price, pe_price = ce_row_final['close'], pe_row_final['close']
                
                return ce_scrip, pe_scrip, ce_price, pe_price, future_price, start_dt
            except (IndexError, KeyError, ValueError, TypeError):
                start_dt -= datetime.timedelta(minutes = 1)

            except Exception as e:
                print('get_Eod_Straddle_strike', e)
                traceback.print_exc()
                start_dt -= datetime.timedelta(minutes = 1)

        return None, None, None, None, None, None

    def get_strangle_strike(self, start_dt, end_dt, om=None, target=None, check_inverted=False, tf=1):
        current_date = start_dt.date()
        future_pl = (self.future_data_pl.filter((pl.col("date_time") >= start_dt) &(pl.col("date_time") <= end_dt)).select(["date_time", "close"]).sort("date_time"))
        
        for row in future_pl.iter_rows(named=True):
            current_dt = row["date_time"]
            if current_dt.date() != current_date:
                break
            try:
                future_price = row["close"]
                one_om = self.get_one_om(future_price)
                target = one_om * om if target is None else target
                
                target_od = self.options_pl.filter(
                    (pl.col('date_time') == current_dt) & 
                    (pl.col('close') >= target * tf)
                ).sort('close').with_columns(pl.col("scrip").cast(pl.Utf8))
                
                ce_series = target_od.filter(pl.col("scrip").str.ends_with("CE")).select("scrip").to_series()
                pe_series = target_od.filter(pl.col("scrip").str.ends_with("PE")).select("scrip").to_series()

                if ce_series.is_empty() or pe_series.is_empty():
                    continue
                ce_scrip = ce_series.item(0)
                pe_scrip = pe_series.item(0)
                
                ce_scrip_list = [ce_scrip,f"{get_strike(ce_scrip) - self.gap}CE",f"{get_strike(ce_scrip) + self.gap}CE"]
                pe_scrip_list = [pe_scrip,f"{get_strike(pe_scrip) - self.gap}PE",f"{get_strike(pe_scrip) + self.gap}PE"]
                
                def get_option_price(scrip):
                    result = (
                        self.options_pl
                        .filter(
                            (pl.col("date_time") == current_dt) &
                            (pl.col("scrip") == scrip)
                        )
                        .select("close")
                    )
                    return result.item() if not result.is_empty() else 0

                call_list_prices = [get_option_price(s) for s in ce_scrip_list]
                put_list_prices  = [get_option_price(s) for s in pe_scrip_list]
                
                call = call_list_prices[0]
                put = put_list_prices[0]
                target_2 = target * 2 * tf
                target_3 = target * 3
                
                min_diff = float("inf")
                required_call, required_put = None, None

                diff = abs(put - call)
                if target_2 <= (put + call) <= target_3 and diff < min_diff:
                    min_diff = diff
                    required_call, required_put = call, put
                    required_call_idx, required_put_idx = 0, 0
                    
                for i in range(1, 3):
                    diff_put = abs(put_list_prices[i] - call)
                    if diff_put < min_diff and target_2 <= (put_list_prices[i] + call) <= target_3:
                        min_diff = diff_put
                        required_call, required_put = call, put_list_prices[i]
                        required_call_idx, required_put_idx = 0, i

                    # Try swapping call
                    diff_call = abs(call_list_prices[i] - put)
                    if diff_call < min_diff and target_2 <= (call_list_prices[i] + put) <= target_3:
                        min_diff = diff_call
                        required_call, required_put = call_list_prices[i], put
                        required_call_idx, required_put_idx = i, 0
                if required_call is None or required_put is None:
                    continue
    
                ce_scrip = ce_scrip_list[required_call_idx]
                pe_scrip = pe_scrip_list[required_put_idx]
                
                
                ce_price = self.options_pl.filter((pl.col("date_time") == current_dt) & (pl.col("scrip") == ce_scrip)).select("close")
                pe_price = self.options_pl.filter((pl.col("date_time") == current_dt) & (pl.col("scrip") == pe_scrip)).select("close")
                                
                ce_price = ce_price.item()
                pe_price = pe_price.item()
                # ---- inverted strike check ----
                if get_strike(ce_scrip) < get_strike(pe_scrip) and check_inverted:
                    return self.get_straddle_strike(current_dt, end_dt)

                return ce_scrip, pe_scrip, ce_price, pe_price, future_price, current_dt

            except (IndexError, KeyError, ValueError, TypeError):
                if current_dt.date() != current_date:
                    break
            except Exception as e:
                print('get_strangle_strike', e)
                traceback.print_exc()
                if current_dt.date() != current_date:
                    break
        return None, None, None, None, None, None    
                
    def get_ut_strike(self, start_dt, end_dt, om=None, target=None):
        
        current_date = start_dt.date()
        
        future_df = (self.future_data_pl.filter(
                (pl.col("date_time") >= start_dt) &
                (pl.col("date_time") <= end_dt)
            ).select(["date_time", "close"]).sort("date_time"))
        
        for rows in future_df.iter_rows(named=True):
            try:
                current_dt = rows["date_time"]
                future_price = rows["close"]
                one_om = self.get_one_om(future_price)
                target = one_om * om if target is None else target
                
                # Filter options_pl using Polars
                target_od = self.options_pl.filter(
                    (pl.col('date_time') == current_dt) & 
                    (pl.col('close') >= target)
                ).sort('close')
                
                if target_od.is_empty():
                    continue
                
                ce_df = target_od.filter(pl.col('scrip').str.ends_with('CE'))
                pe_df = target_od.filter(pl.col('scrip').str.ends_with('PE'))
                
                if ce_df.is_empty() or pe_df.is_empty():
                    continue
                ce_scrip, ce_price = ce_df.select(["scrip", "close"]).row(0)
                pe_scrip, pe_price = pe_df.select(["scrip", "close"]).row(0)
                
                

                return ce_scrip, pe_scrip, ce_price, pe_price, future_price, current_dt
            except (IndexError, KeyError, ValueError, TypeError):
                if current_dt.date() != current_date: break
            except Exception as e:
                print('get_straddle_strike', e)
                traceback.print_exc()
                if current_dt.date() != current_date: break

    def _get_strike(self, start_dt, end_dt, om=None, target=None, check_inverted=False, tf=1, only=None, obove_target_only=False, SDroundoff=False):
        
        if '%' in str(om) or obove_target_only:
            
            if '%' in str(om):
                om_precent = float(om.replace('%', ''))
                future_price = self.future_data_pl.select(pl.col("close")).item(0)  # Get the first close price as future price
                one_om = self.get_one_om(future_price)
                target = one_om*om_precent/100

            ce_scrip, pe_scrip, ce_price, pe_price, future_price, start_dt = self.get_ut_strike(start_dt, end_dt, om=om, target=target)  
        else:
            if 'SD' in str(om).upper():
                sd = float(om.upper().replace(' ', '').replace('SD', ''))
                om = None
            else:
                sd = 0
                om = float(om) if om else om

            if (om is None or om <= 0) and target is None:
                ce_scrip, pe_scrip, ce_price, pe_price, future_price, start_dt = self.get_straddle_strike(start_dt, end_dt, sd=sd, SDroundoff=SDroundoff)
            else:
                ce_scrip, pe_scrip, ce_price, pe_price, future_price, start_dt = self.get_strangle_strike(start_dt, end_dt, om=om, target=target, check_inverted=check_inverted, tf=tf)
                
        if only is None:
            return ce_scrip, pe_scrip, ce_price, pe_price, future_price, start_dt
        else:
            if only == "CE":
                return ce_scrip, ce_price, future_price, start_dt
            elif only == "PE":
                return pe_scrip, pe_price, future_price, start_dt

    def sl_check_by_given_data(self, scrip_df, o=None, sl=0, intra_sl=0, sl_price=None, target_price=None, from_candle_close=False, orderside='SELL', from_next_minute=True, with_ohlc=False, pl_with_slipage=True, per_minute_mtm=False, roundtick=False, pandas=True):
        """
        # Pandas in → Pandas out (default)
        sl_check_by_given_data(df)

        # Polars in → Polars out
        sl_check_by_given_data(pl_df, pandas=False)

        # Pandas with MTM
        sl_check_by_given_data(df, per_minute_mtm=True)

        # Polars MTM (ultra fast)
        sl_check_by_given_data(pl_df, per_minute_mtm=True, pandas=False)

        """
        sl_flag, intra_sl_flag, target_flag, exit_time, pnl = False, False, False, '', 0
        try:
            if pandas:
                if scrip_df.empty: raise DataEmptyError
                pl_df = pl.from_pandas(scrip_df.reset_index())
                pl_df = pl_df.with_columns(pl.col('date_time').str.strptime(pl.Datetime, strict=False).alias("date_time"))
            else:
                if scrip_df.is_empty(): raise DataEmptyError
                pl_df = scrip_df
            
            scrip_df = scrip_df.with_columns([
                pl.when(pl.col("date_time").dt.time() == self.meta_start_time)
                .then(pl.col("close"))
                .otherwise(pl.col("high"))
                .alias("high"),

                pl.when(pl.col("date_time").dt.time() == self.meta_start_time)
                .then(pl.col("close"))
                .otherwise(pl.col("low"))
                .alias("low"),
            ])

            
            o = pl_df.select("close").item(0) if o is None else o
            slipage = self.Cal_slipage(o) if pl_with_slipage else 0

            if from_next_minute: pl_df = pl_df.slice(1)
            if pl_df.is_empty(): raise DataEmptyError

            h = pl_df.select(pl.col("high").max()).item()
            l = pl_df.select(pl.col("low").min()).item()
            c = pl_df.select(pl.col("close").last()).item()
            
            if orderside == 'SELL':
                sl_price_val = (((100 + sl) / 100) * o if sl_price is None else sl_price) if (sl or sl_price) else (h + 1)
                intra_sl_price = ((100 + intra_sl) / 100) * o if intra_sl else (h + 1)
                target_price = target_price if target_price is not None else (l - 1)
            else:
                sl_price_val = (((100 - sl) / 100) * o if sl_price is None else sl_price) if (sl or sl_price) else (l - 1)
                intra_sl_price = ((100 - intra_sl) / 100) * o if intra_sl else (l - 1)
                target_price = target_price if target_price is not None else (h + 1)

            if roundtick or self.market == 'MCX':
                sl_price_val = self.round_to_ticksize(sl_price_val, orderside, 'STOPLOSS')
                intra_sl_price = self.round_to_ticksize(intra_sl_price, orderside, 'STOPLOSS')
                target_price = self.round_to_ticksize(target_price, orderside, 'TARGET')

            if orderside == 'SELL':
                sl_col = "close" if from_candle_close else "high"
                exit_cond = (
                    (pl.col("high") >= intra_sl_price) |
                    (pl.col(sl_col) >= sl_price_val) |
                    (pl.col("low") <= target_price))
            else:
                sl_col = "close" if from_candle_close else "low"
                exit_cond = (
                    (pl.col("low") <= intra_sl_price) |
                    (pl.col(sl_col) <= sl_price_val) |
                    (pl.col("high") >= target_price))
            
            combined_mask = pl_df.filter(exit_cond).sort("date_time")

            if not combined_mask.is_empty():
                exit_row = combined_mask.row(0, named=True)
                exit_time = exit_row["date_time"]


                if orderside == 'SELL':
                    if exit_row['high'] >= intra_sl_price:
                        sl_flag, intra_sl_flag = True, True
                        exit_price = intra_sl_price
                    elif (exit_row['close'] if from_candle_close else exit_row['high']) >= sl_price_val:
                        sl_flag = True
                        exit_price = exit_row['close'] if from_candle_close else sl_price_val 
                    elif exit_row['low'] <= target_price:
                        target_flag = True
                        exit_price = target_price
                elif orderside == 'BUY':
                    if exit_row['low'] <= intra_sl_price:
                        sl_flag, intra_sl_flag = True, True
                        exit_price = intra_sl_price
                    elif (exit_row['close'] if from_candle_close else exit_row['low']) <= sl_price_val:
                        sl_flag = True
                        exit_price = exit_row['close'] if from_candle_close else sl_price_val
                    elif exit_row['high'] >= target_price:
                        target_flag = True
                        exit_price = target_price
            else:
                exit_price = c

            if sl_flag and exit_time.time() == self.meta_start_time:
                exit_price = scrip_df.filter(pl.col("date_time") == exit_time).select("close").item()

            pnl = (exit_price - o) if orderside == 'BUY' else (o - exit_price)
            pnl = round(pnl - slipage, 2)
            
            
            if per_minute_mtm:
                scrip_df = scrip_df.to_pandas()
                scrip_df.set_index('date_time', inplace=True)
                if exit_time:
                    scrip_df = scrip_df.loc[scrip_df.index <= exit_time]

                per_minute_mtm_series = o - scrip_df['close'] if orderside == 'SELL' else scrip_df['close'] - o
                per_minute_mtm_series = per_minute_mtm_series - slipage
                per_minute_mtm_series.iloc[-1] = pnl
                if not pandas:
                    per_minute_mtm_series = pl.Series(per_minute_mtm_series)
            
                

        except DataEmptyError:
            sl_flag, intra_sl_flag, target_flag, exit_time, pnl = False, False, False, '', 0
            o, h, l, c = '', '', '', ''
            per_minute_mtm_series = pd.Series() if pandas else pl.Series()
            sl_price_val = ''
        except Exception as e:
            print('sl_check_single_leg', e)
            traceback.print_exc()
            sl_flag, intra_sl_flag, target_flag, exit_time, pnl = False, False, False, '', 0
            o, h, l, c = '', '', '', ''
            per_minute_mtm_series = pd.Series() if pandas else pl.Series()
            sl_price_val = ''

        sl_price = sl_price_val if (sl or sl_price) else ''

        if with_ohlc:
            ohlc_data = (o, h, l, c, sl_price)
            if per_minute_mtm:
                return (*ohlc_data, exit_time, per_minute_mtm_series)
            return (*ohlc_data, sl_flag, intra_sl_flag, target_flag, exit_time, pnl)
        if per_minute_mtm:
            return (exit_time, per_minute_mtm_series)
        return (sl_price, sl_flag, intra_sl_flag, target_flag, exit_time, pnl)

    def _sl_check_single_leg(self, start_dt, end_dt, scrip, o=None, sl=0, intra_sl=0, sl_price=None, target_price=None, from_candle_close=False, orderside='SELL', from_next_minute=True, with_ohlc=False, pl_with_slipage=True, per_minute_mtm=False, roundtick=False, pandas=True):
        """
        # Pandas in → Pandas out (default)
        sl_check_single_leg(start_dt, end_dt, scrip)
        
        # Polars in → Polars out
        sl_check_single_leg(start_dt, end_dt, scrip, pandas=False)
        
        # Pandas with MTM
        sl_check_single_leg(start_dt, end_dt, scrip, per_minute_mtm=True)
        
        # Polars MTM (ultra fast)
        sl_check_single_leg(start_dt, end_dt, scrip, per_minute_mtm=True, pandas=False)             
        """
        sl_flag, intra_sl_flag, target_flag, exit_time, pnl = False, False, False, '', 0

        try:
            scrip_df:pl.DataFrame = self.get_single_leg_data(start_dt, end_dt, scrip, pandas=False)
            if scrip_df.is_empty(): raise DataEmptyError
            
            scrip_df = scrip_df.with_columns([
                pl.when(pl.col("date_time").dt.time() == self.meta_start_time)
                .then(pl.col("close"))
                .otherwise(pl.col("high"))
                .alias("high"),

                pl.when(pl.col("date_time").dt.time() == self.meta_start_time)
                .then(pl.col("close"))
                .otherwise(pl.col("low"))
                .alias("low"),
            ])
            
            o = scrip_df.select("close").item(0) if o is None else o
            slipage = self.Cal_slipage(o) if pl_with_slipage else 0

            if from_next_minute: scrip_df = scrip_df.slice(1)
            if scrip_df.is_empty(): raise DataEmptyError
            
            h = scrip_df.select(pl.col("high").max()).item()
            l = scrip_df.select(pl.col("low").min()).item()
            c = scrip_df.select(pl.col("close").last()).item()
            
            if orderside == 'SELL':
                sl_price_val = (((100 + sl) / 100) * o if sl_price is None else sl_price) if (sl or sl_price) else (h + 1)
                intra_sl_price = ((100 + intra_sl) / 100) * o if intra_sl else (h + 1)
                target_price = target_price if target_price is not None else (l - 1)                
            else:
                sl_price_val = (((100 - sl) / 100) * o if sl_price is None else sl_price) if (sl or sl_price) else (l - 1)
                intra_sl_price = ((100 - intra_sl) / 100) * o if intra_sl else (l - 1)
                target_price = target_price if target_price is not None else (h + 1)
            
            if roundtick or self.market == 'MCX':
                sl_price_val = self.round_to_ticksize(sl_price_val, orderside, 'STOPLOSS')
                intra_sl_price = self.round_to_ticksize(intra_sl_price, orderside, 'STOPLOSS')
                target_price = self.round_to_ticksize(target_price, orderside, 'TARGET')

            if orderside == 'SELL':
                sl_col = "close" if from_candle_close else "high"
                exit_expr = ((pl.col("high") >= intra_sl_price) | (pl.col(sl_col) >= sl_price_val) | (pl.col("low") <= target_price))
            else:
                sl_col = "close" if from_candle_close else "low"
                exit_expr = ((pl.col("low") <= intra_sl_price) | (pl.col(sl_col) <= sl_price_val) | (pl.col("high") >= target_price))
            
            combined_mask = scrip_df.filter(exit_expr).sort("date_time")            
        
            if not combined_mask.is_empty():
                exit_row = combined_mask.row(0, named=True)
                exit_time = exit_row['date_time']

                if orderside == 'SELL':
                    if exit_row['high'] >= intra_sl_price:
                        sl_flag, intra_sl_flag = True, True
                        exit_price = intra_sl_price
                    elif (exit_row['close'] if from_candle_close else exit_row['high']) >= sl_price_val:
                        sl_flag = True
                        exit_price = exit_row['close'] if from_candle_close else sl_price_val 
                    elif exit_row['low'] <= target_price:
                        target_flag = True
                        exit_price = target_price
                elif orderside == 'BUY':
                    if exit_row['low'] <= intra_sl_price:
                        sl_flag, intra_sl_flag = True, True
                        exit_price = intra_sl_price
                    elif (exit_row['close'] if from_candle_close else exit_row['low']) <= sl_price_val:
                        sl_flag = True
                        exit_price = exit_row['close'] if from_candle_close else sl_price_val
                    elif exit_row['high'] >= target_price:
                        target_flag = True
                        exit_price = target_price
            else:
                exit_price = c

            if sl_flag and exit_time.time() == self.meta_start_time:
                exit_price = scrip_df.filter(pl.col("date_time") == exit_time).select("close").item()
            
            pnl = (exit_price - o) if orderside == 'BUY' else (o - exit_price)
            pnl = round(pnl - slipage, 2)

            # ---------- PER MINUTE MTM ----------
            
            if per_minute_mtm:
                scrip_df = scrip_df.to_pandas()
                scrip_df.set_index('date_time', inplace=True)
                if exit_time:
                    scrip_df = scrip_df.loc[scrip_df.index <= exit_time]

                per_minute_mtm_series = o - scrip_df['close'] if orderside == 'SELL' else scrip_df['close'] - o
                per_minute_mtm_series = per_minute_mtm_series - slipage
                per_minute_mtm_series.iloc[-1] = pnl
                if not pandas:
                    per_minute_mtm_series = pl.Series(per_minute_mtm_series)
            
        except DataEmptyError:
            sl_flag, intra_sl_flag, target_flag, exit_time, pnl = False, False, False, '', 0
            o, h, l, c = '', '', '', ''
            per_minute_mtm_series = pd.Series() if pandas else pl.Series()
            sl_price_val = ''
        except Exception as e:
            print('sl_check_single_leg', e)
            traceback.print_exc()
            sl_flag, intra_sl_flag, target_flag, exit_time, pnl = False, False, False, '', 0
            o, h, l, c = '', '', '', ''
            per_minute_mtm_series = pd.Series() if pandas else pl.Series()
            sl_price_val = ''

        sl_price = sl_price_val if (sl or sl_price) else ''

        if with_ohlc:
            ohlc_data = (o, h, l, c, sl_price)
            if per_minute_mtm:
                return (*ohlc_data, exit_time, per_minute_mtm_series)
            
            return (*ohlc_data, sl_flag, intra_sl_flag, target_flag, exit_time, pnl)
        if per_minute_mtm:
            return (exit_time, per_minute_mtm_series)
        return (sl_price, sl_flag, intra_sl_flag, target_flag, exit_time, pnl)

    def _sl_check_combine_leg(self, start_dt, end_dt, ce_scrip, pe_scrip, o=None, sl=0, intra_sl=0, sl_price=None, intra_sl_price=None, target_price=None, orderside='SELL', from_next_minute=True, with_ohlc=False, pl_with_slipage=True, per_minute_mtm=False, roundtick=False, pandas=True):
        """
            # Pandas in → Pandas out (default)
            sl_check_combine_leg(start_dt, end_dt, ce_scrip, pe_scrip)
            
            # Polars in → Polars out
            sl_check_combine_leg(start_dt, end_dt, ce_scrip, pe_scrip, pandas=False)
            
            # Pandas with MTM
            sl_check_combine_leg(start_dt, end_dt, ce_scrip, pe_scrip, per_minute_mtm=True)
            
            # Polars MTM (ultra fast)
            sl_check_combine_leg(start_dt, end_dt, ce_scrip, pe_scrip, per_minute_mtm=True, pandas=False)             
        """
     
        sl_flag, intra_sl_flag, target_flag, exit_time, pnl = False, False, False, '', 0

        try:
            scrip_df = self.get_straddle_data(start_dt, end_dt, ce_scrip, pe_scrip, pandas=False)
            if scrip_df.is_empty(): raise DataEmptyError
            
            scrip_df = scrip_df.with_columns([
                pl.when(pl.col("date_time").dt.time() == self.meta_start_time)
                .then(pl.col("close"))
                .otherwise(pl.col("high"))
                .alias("high"),

                pl.when(pl.col("date_time").dt.time() == self.meta_start_time)
                .then(pl.col("close"))
                .otherwise(pl.col("low"))
                .alias("low"),
            ])
            
            o = scrip_df.select("close").item(0) if o is None else o
            slipage = self.Cal_slipage(o) if pl_with_slipage else 0
            
            if from_next_minute: scrip_df = scrip_df.slice(1)
            if scrip_df.is_empty(): raise DataEmptyError
            h, l, cl, ch, c = (scrip_df.select([
                    pl.col("high").max().alias("h"),
                    pl.col("low").min().alias("l"),
                    pl.col("close").min().alias("cl"),
                    pl.col("close").max().alias("ch"),
                    pl.col("close").last().alias("c"),]).row(0))

            if orderside == 'SELL':
                sl_price_val = (((100 + sl) / 100) * o if sl_price is None else sl_price) if (sl or sl_price) else (ch + 1)
                intra_sl_price_val = (((100 + intra_sl) / 100) * o if intra_sl_price is None else intra_sl_price) if (intra_sl or intra_sl_price) else (h + 1)
                target_price = target_price if target_price is not None else (cl - 1)
            
            elif orderside == 'BUY':
                sl_price_val = (((100 - sl) / 100) * o if sl_price is None else sl_price) if (sl or sl_price) else (cl - 1)
                intra_sl_price_val = (((100 - intra_sl) / 100) * o if intra_sl_price is None else intra_sl_price) if (intra_sl or intra_sl_price) else (l - 1)
                target_price = target_price if target_price is not None else (ch + 1)
            
            if roundtick or self.market == 'MCX':
                sl_price_val = self.round_to_ticksize(sl_price_val, orderside, 'STOPLOSS')
                intra_sl_price_val = self.round_to_ticksize(intra_sl_price_val, orderside, 'STOPLOSS')
                target_price = self.round_to_ticksize(target_price, orderside, 'TARGET')

            if orderside == 'SELL':
                exit_expr = ((pl.col("high") >= intra_sl_price_val) | (pl.col("close") >= sl_price_val) | (pl.col("close") <= target_price))
            else:
                exit_expr = ((pl.col("low") <= intra_sl_price_val) | (pl.col("close") <= sl_price_val) | (pl.col("close") >= target_price))
            
            combined_mask = scrip_df.filter(exit_expr).sort("date_time")
            if not combined_mask.is_empty():
                exit_row = combined_mask.row(0, named=True)
                exit_time = exit_row['date_time']
                if orderside == 'SELL':
                    if exit_row['high'] >= intra_sl_price_val:
                        sl_flag, intra_sl_flag = True, True
                        exit_price = intra_sl_price_val
                    elif exit_row['close'] >= sl_price_val:
                        sl_flag = True
                        exit_price = exit_row['close']
                    elif exit_row['close'] <= target_price:
                        target_flag = True
                        exit_price = exit_row['close']
                elif orderside == 'BUY':
                    if exit_row['low'] <= intra_sl_price_val:
                        sl_flag, intra_sl_flag = True, True
                        exit_price = intra_sl_price_val
                    elif exit_row['close'] <= sl_price_val:
                        sl_flag = True
                        exit_price = exit_row['close']
                    elif exit_row['close'] >= target_price:
                        target_flag = True
                        exit_price = exit_row['close']
            else:
                exit_price = c
                
            if sl_flag and exit_time.time() == self.meta_start_time:
                exit_price = scrip_df.filter(pl.col("date_time") == exit_time).select("close").item()
                
            pnl = (exit_price - o) if orderside == 'BUY' else (o - exit_price)
            pnl = round(pnl - slipage, 2)
            
            if per_minute_mtm:
                scrip_df = scrip_df.to_pandas()
                scrip_df.set_index('date_time', inplace=True)
                if exit_time:
                    scrip_df = scrip_df.loc[scrip_df.index <= exit_time]
                per_minute_mtm_series = o - scrip_df['close'] if orderside == 'SELL' else scrip_df['close'] - o
                per_minute_mtm_series = per_minute_mtm_series - slipage
                per_minute_mtm_series.iloc[-1] = pnl
                if not pandas:
                    per_minute_mtm_series = pl.Series(per_minute_mtm_series)
            
        except DataEmptyError:
            sl_flag, intra_sl_flag, target_flag, exit_time, pnl = False, False, False, '', 0
            o, h, l, c = '', '', '', ''
            per_minute_mtm_series = pd.Series() if pandas else pl.Series()
            sl_price_val, intra_sl_price_val = '', ''
        except Exception as e:
            print('sl_check_combine_leg', e)
            traceback.print_exc()
            sl_flag, intra_sl_flag, target_flag, exit_time, pnl = False, False, False, '', 0
            o, h, l, c = '', '', '', ''
            per_minute_mtm_series = pd.Series() if pandas else pl.Series()
            sl_price_val, intra_sl_price_val = '', ''
        sl_price = sl_price_val if (sl or sl_price) else ''
        intra_sl_price = intra_sl_price_val if (intra_sl or intra_sl_price) else ''
        
        if with_ohlc:
            ohlc_data = (o, h, l, c, sl_price, intra_sl_price)
            if per_minute_mtm:
                return (*ohlc_data, exit_time, per_minute_mtm_series)
            return (*ohlc_data, sl_flag, intra_sl_flag, target_flag, exit_time, pnl)
        if per_minute_mtm:
            return (exit_time, per_minute_mtm_series)
        return (sl_price, intra_sl_price, sl_flag, intra_sl_flag, target_flag, exit_time, pnl)
             
    def _sl_range_check_combine_leg(self, start_dt, end_dt, ce_scrip, pe_scrip, lower_range, upper_range, intra_lower_range, intra_upper_range, straddle_strike, orderside='SELL', from_next_minute=True, with_ohlc=False, pl_with_slipage=True, per_minute_mtm=False, eod_modify=False, range_sl=None, intra_range_sl=None, is_on_synthetic=False, need_day_wise_mtm=False):
        sl_flag, intra_sl_flag, exit_time, pnl = False, False, '', 0
        day_wise_mtm, day_wise_mtm2 = {}, {}

        try:
            # Get pandas DataFrame and convert to Polars
            scrip_df:pl.DataFrame = self.get_straddle_data(start_dt, end_dt, ce_scrip, pe_scrip)
            if scrip_df.is_empty(): raise DataEmptyError
            
            o = scrip_df['close'][0]
            if from_next_minute: 
                scrip_df = scrip_df.slice(1)
            if scrip_df.height == 0: raise DataEmptyError

            h = scrip_df['high'].max()
            l = scrip_df['low'].min()
            cl = scrip_df['close'].min()
            ch = scrip_df['close'].max()
            c = scrip_df['close'][-1]
            slipage = self.Cal_slipage(o) if pl_with_slipage else 0
            
            # Convert to list for iteration
            date_times = scrip_df['date_time'].to_list()
            closes = scrip_df['close'].to_list()
            highs = scrip_df['high'].to_list()
            lows = scrip_df['low'].to_list()
            
            dstart, dstartprice = date_times[0], o
            current_dt = date_times[0]
            exit_price = None
            
            # Pre-filter options_pl for straddle strike CE and PE for faster lookups
            ce_scrip_std = f"{straddle_strike}CE"
            pe_scrip_std = f"{straddle_strike}PE"
            ce_std_data = self.options_pl.filter(pl.col('scrip') == ce_scrip_std)
            pe_std_data = self.options_pl.filter(pl.col('scrip') == pe_scrip_std)
            
            # Create lookup dictionaries for faster access
            ce_std_dict = {row['date_time']: row for row in ce_std_data.iter_rows(named=True)}
            pe_std_dict = {row['date_time']: row for row in pe_std_data.iter_rows(named=True)}
            
            # Create future_data lookup dictionary from Polars
            future_dict = {row['date_time']: row for row in self.future_data_pl.iter_rows(named=True)}
            
            for idx in range(len(date_times) - 1):
                current_dt = date_times[idx]
                current_close = closes[idx]
                current_high = highs[idx]
                current_low = lows[idx]

                try:
                    ce_std_data_row = ce_std_dict.get(current_dt)
                    pe_std_data_row = pe_std_dict.get(current_dt)
                    future_data_row = future_dict.get(current_dt)
                    
                    if ce_std_data_row is None or pe_std_data_row is None or future_data_row is None:
                        continue
                    
                    if is_on_synthetic:
                        future_high = straddle_strike + ce_std_data_row['high'] - pe_std_data_row['low']
                        future_low = straddle_strike + ce_std_data_row['low'] - pe_std_data_row['high']
                        future_close = straddle_strike + ce_std_data_row['close'] - pe_std_data_row['close']
                    else:
                        future_high, future_low, future_close = future_data_row['high'], future_data_row['low'], future_data_row['close']

                    if (current_dt.time() != self.meta_start_time) and (intra_upper_range <= future_high or future_low <= intra_lower_range):
                        sl_flag = True
                        intra_sl_flag = True
                        exit_time = current_dt
                        exit_price = current_high if orderside == 'SELL' else current_low
                        break
                
                    elif upper_range <= future_close or future_close <= lower_range:
                        sl_flag = True
                        exit_time = current_dt
                        exit_price = current_close
                        break
                except:
                    pass

                if eod_modify and current_dt.date() != date_times[idx + 1].date():
                    try:
                        _, _, std_tce_price, std_tpe_price, _, _ = self.get_EOD_straddle_strike(current_dt.date())
                        lower_range, upper_range, intra_lower_range, intra_upper_range = self.get_sl_range(straddle_strike, std_tce_price+std_tpe_price, range_sl, intra_range_sl)
                    except:
                        pass

                if need_day_wise_mtm and current_dt.date() != date_times[idx + 1].date():
                    dend, dendprice = current_dt, current_close
                    dendpnl = (dstartprice - dendprice) if (orderside == 'SELL') else (dendprice - dstartprice)
                    day_wise_mtm[dend.date()] = day_wise_mtm.get(dend.date(), 0) + dendpnl
                    day_wise_mtm2[(dstart, dend)] = dendpnl
                    dstart, dstartprice = dend, dendprice
                    
            if not sl_flag:
                exit_price = c

            if need_day_wise_mtm:
                dend = current_dt
                dendprice = exit_price if sl_flag else c
                dendpnl = (dstartprice - dendprice) if (orderside == 'SELL') else (dendprice - dstartprice)
                day_wise_mtm[dend.date()] = day_wise_mtm.get(dend.date(), 0) + dendpnl - slipage
                day_wise_mtm2[(dstart, dend)] = dendpnl - slipage

            pnl = (exit_price - o) if orderside == 'BUY' else (o - exit_price)
            pnl = round(pnl - slipage, 2)

            if per_minute_mtm:
                # Filter by exit_time if present
                if exit_time:
                    scrip_df = scrip_df.filter(pl.col('date_time') <= exit_time)
                
                if orderside == 'SELL':
                    mtm_values = o - scrip_df['close']
                else:
                    mtm_values = scrip_df['close'] - o
                
                mtm_values = mtm_values - slipage
                
                per_minute_mtm_series = pd.Series(
                    mtm_values.to_list(),
                    index=pd.to_datetime(scrip_df['date_time'].to_list())
                )
                per_minute_mtm_series.iloc[-1] = pnl

        except DataEmptyError:
            sl_flag, intra_sl_flag, exit_time, pnl = False, False, '', 0
            o, h, l, c = '', '', '', ''
            per_minute_mtm_series = pd.Series()
            day_wise_mtm, day_wise_mtm2 = {}, {}
        except Exception as e:
            print('sl_check_combine_leg', e)
            traceback.print_exc()
            sl_flag, intra_sl_flag, exit_time, pnl = False, False, '', 0
            o, h, l, c = '', '', '', ''
            per_minute_mtm_series = pd.Series()
            day_wise_mtm, day_wise_mtm2 = {}, {}

        if with_ohlc:
            ohlc_data = (o, h, l, c)
            if per_minute_mtm:
                return (*ohlc_data, exit_time, per_minute_mtm_series)
            else:
                if need_day_wise_mtm:
                    return (*ohlc_data, sl_flag, intra_sl_flag, exit_time, day_wise_mtm, day_wise_mtm2, pnl)
                else:
                    return (*ohlc_data, sl_flag, intra_sl_flag, exit_time, pnl)
        else:
            if per_minute_mtm:
                return (exit_time, per_minute_mtm_series)
            else:
                if need_day_wise_mtm:
                    return (sl_flag, intra_sl_flag, exit_time, day_wise_mtm, day_wise_mtm2, pnl)
                else:
                    return (sl_flag, intra_sl_flag, exit_time, pnl)
                
    def _sl_range_trail_check_combine_leg(self, start_dt, end_dt, ce_scrip, pe_scrip, lower_range, upper_range, intra_lower_range, intra_upper_range, straddle_strike, straddle_price, sl, intra_sl, orderside='SELL', from_next_minute=True, with_ohlc=False, pl_with_slipage=True, per_minute_mtm=False, eod_modify=False, range_sl=None, intra_range_sl=None, is_on_synthetic=False, need_day_wise_mtm=False):
        sl_flag, intra_sl_flag, exit_time, pnl = False, False, '', 0
        day_wise_mtm, day_wise_mtm2 = {}, {}

        try:
            # Get pandas DataFrame and convert to Polars
            scrip_df:pl.DataFrame = self.get_straddle_data(start_dt, end_dt, ce_scrip, pe_scrip)
            if scrip_df.is_empty(): raise DataEmptyError
            
            o = scrip_df['close'][0]
            
            if from_next_minute:
                scrip_df = scrip_df.slice(1)
            if scrip_df.height == 0: raise DataEmptyError

            h = scrip_df['high'].max()
            l = scrip_df['low'].min()
            cl = scrip_df['close'].min()
            ch = scrip_df['close'].max()
            c = scrip_df['close'][-1]
            slipage = self.Cal_slipage(o) if pl_with_slipage else 0
            
            # Convert to list for iteration
            date_times = scrip_df['date_time'].to_list()
            closes = scrip_df['close'].to_list()
            highs = scrip_df['high'].to_list()
            lows = scrip_df['low'].to_list()
            
            dstart, dstartprice = date_times[0], o
            current_dt = date_times[0]
            exit_price = None
            
            # Pre-filter options_pl for straddle strike CE and PE for faster lookups
            ce_scrip_std = f"{straddle_strike}CE"
            pe_scrip_std = f"{straddle_strike}PE"
            ce_std_data = self.options_pl.filter(pl.col('scrip') == ce_scrip_std)
            pe_std_data = self.options_pl.filter(pl.col('scrip') == pe_scrip_std)
            
            # Create lookup dictionaries for faster access
            ce_std_dict = {row['date_time']: row for row in ce_std_data.iter_rows(named=True)}
            pe_std_dict = {row['date_time']: row for row in pe_std_data.iter_rows(named=True)}
            
            # Create future_data lookup dictionary from Polars
            future_dict = {row['date_time']: row for row in self.future_data_pl.iter_rows(named=True)}
            
            for idx in range(len(date_times) - 1):
                current_dt = date_times[idx]
                current_close = closes[idx]
                current_high = highs[idx]
                current_low = lows[idx]

                try:
                    ce_std_data_row = ce_std_dict.get(current_dt)
                    pe_std_data_row = pe_std_dict.get(current_dt)
                    future_data_row = future_dict.get(current_dt)
                    
                    if ce_std_data_row is None or pe_std_data_row is None or future_data_row is None:
                        continue
                    
                    _, _, temp_std_ce_price, temp_std_pe_price, _, temp_std_current_dt = self.get_strike(current_dt, current_dt, om=0)

                    if (temp_std_current_dt == current_dt) and ((temp_std_ce_price + temp_std_pe_price) < straddle_price):
                        straddle_price = temp_std_ce_price + temp_std_pe_price
                        lower_range, upper_range, intra_lower_range, intra_upper_range = self.get_sl_range(straddle_strike, straddle_price, sl, intra_sl)
                    
                    if is_on_synthetic:
                        future_high = straddle_strike + ce_std_data_row['high'] - pe_std_data_row['low']
                        future_low = straddle_strike + ce_std_data_row['low'] - pe_std_data_row['high']
                        future_close = straddle_strike + ce_std_data_row['close'] - pe_std_data_row['close']
                    else:
                        future_high, future_low, future_close = future_data_row['high'], future_data_row['low'], future_data_row['close']

                    if (current_dt.time() != self.meta_start_time) and (intra_upper_range <= future_high or future_low <= intra_lower_range):
                        sl_flag = True
                        intra_sl_flag = True
                        exit_time = current_dt
                        exit_price = current_high if orderside == 'SELL' else current_low
                        break
                
                    elif upper_range <= future_close or future_close <= lower_range:
                        sl_flag = True
                        exit_time = current_dt
                        exit_price = current_close
                        break
                except:
                    pass

                if eod_modify and current_dt.date() != date_times[idx + 1].date():
                    try:
                        _, _, std_tce_price, std_tpe_price, _, _ = self.get_EOD_straddle_strike(current_dt.date())
                        lower_range, upper_range, intra_lower_range, intra_upper_range = self.get_sl_range(straddle_strike, std_tce_price+std_tpe_price, range_sl, intra_range_sl)
                    except:
                        pass

                if need_day_wise_mtm and current_dt.date() != date_times[idx + 1].date():
                    dend, dendprice = current_dt, current_close
                    dendpnl = (dstartprice - dendprice) if (orderside == 'SELL') else (dendprice - dstartprice)
                    day_wise_mtm[dend.date()] = day_wise_mtm.get(dend.date(), 0) + dendpnl
                    day_wise_mtm2[(dstart, dend)] = dendpnl
                    dstart, dstartprice = dend, dendprice
                    
            if not sl_flag:
                exit_price = c

            if need_day_wise_mtm:
                dend = current_dt
                dendprice = exit_price if sl_flag else c
                dendpnl = (dstartprice - dendprice) if (orderside == 'SELL') else (dendprice - dstartprice)
                day_wise_mtm[dend.date()] = day_wise_mtm.get(dend.date(), 0) + dendpnl - slipage
                day_wise_mtm2[(dstart, dend)] = dendpnl - slipage

            pnl = (exit_price - o) if orderside == 'BUY' else (o - exit_price)
            pnl = round(pnl - slipage, 2)

            if per_minute_mtm:
                # Filter by exit_time if present
                if exit_time:
                    scrip_df = scrip_df.filter(pl.col('date_time') <= exit_time)
                
                if orderside == 'SELL':
                    mtm_values = o - scrip_df['close']
                else:
                    mtm_values = scrip_df['close'] - o
                
                mtm_values = mtm_values - slipage
                
                per_minute_mtm_series = pd.Series(
                    mtm_values.to_list(),
                    index=pd.to_datetime(scrip_df['date_time'].to_list())
                )
                per_minute_mtm_series.iloc[-1] = pnl

        except DataEmptyError:
            sl_flag, intra_sl_flag, exit_time, pnl = False, False, '', 0
            o, h, l, c = '', '', '', ''
            per_minute_mtm_series = pd.Series()
            day_wise_mtm, day_wise_mtm2 = {}, {}
        except Exception as e:
            print('sl_check_combine_leg', e)
            traceback.print_exc()
            sl_flag, intra_sl_flag, exit_time, pnl = False, False, '', 0
            o, h, l, c = '', '', '', ''
            per_minute_mtm_series = pd.Series()
            day_wise_mtm, day_wise_mtm2 = {}, {}

        if with_ohlc:
            ohlc_data = (o, h, l, c)
            if per_minute_mtm:
                return (*ohlc_data, exit_time, per_minute_mtm_series)
            else:
                if need_day_wise_mtm:
                    return (*ohlc_data, sl_flag, intra_sl_flag, exit_time, day_wise_mtm, day_wise_mtm2, pnl)
                else:
                    return (*ohlc_data, sl_flag, intra_sl_flag, exit_time, pnl)
        else:
            if per_minute_mtm:
                return (exit_time, per_minute_mtm_series)
            else:
                if need_day_wise_mtm:
                    return (sl_flag, intra_sl_flag, exit_time, day_wise_mtm, day_wise_mtm2, pnl)
                else:
                    return (sl_flag, intra_sl_flag, exit_time, pnl)                

    def _decay_check_single_leg(self, start_dt, end_dt, scrip, decay=None, decay_price=None, from_candle_close=False, orderside='SELL', from_next_minute=True, with_ohlc=False, roundtick=False):

        decay_flag, decay_time = False, ''
        
        try:
            scrip_df:pl.DataFrame = self.get_single_leg_data(start_dt, end_dt, scrip, pandas=True).copy()
            if scrip_df.empty: raise DataEmptyError
            
            scrip_df = scrip_df.with_columns([
                pl.when(pl.col("date_time").dt.time() == self.meta_start_time)
                .then(pl.col("close"))
                .otherwise(pl.col("high"))
                .alias("high"),

                pl.when(pl.col("date_time").dt.time() == self.meta_start_time)
                .then(pl.col("close"))
                .otherwise(pl.col("low"))
                .alias("low"),
            ])

            o = scrip_df.select(pl.col("close")).row(0)[0]

            if from_next_minute: scrip_df = scrip_df.slice(1)
            if scrip_df.is_empty(): raise DataEmptyError
                
            h = scrip_df.select(pl.col("high").max()).item()
            l = scrip_df.select(pl.col("low").min()).item()
            c = scrip_df.select(pl.col("close")).row(-1)[0]

            if decay == 0 or decay_price == -1:
                decay_price = o
                decay_flag = True
                decay_time = start_dt
            else:
                if orderside == 'SELL':
                    decay_price = ((100 - decay)/100) * o if decay_price is None else decay_price
                    
                    if roundtick or self.market == 'MCX':
                        decay_price = self.round_to_ticksize(decay_price, orderside, 'DECAY')
                    
                    trigger_col = "close" if from_candle_close else "low"
                    mask_decay = pl.col(trigger_col) <= decay_price
                elif orderside == 'BUY':
                    decay_price = ((100 + decay)/100) * o if decay_price is None else decay_price
                    
                    if roundtick or self.market == 'MCX':
                        decay_price = self.round_to_ticksize(decay_price, orderside, 'DECAY')
                    
                    trigger_col = "close" if from_candle_close else "high"
                    mask_decay = pl.col(trigger_col) >= decay_price
                
                combined_mask = scrip_df.filter(mask_decay).sort("date_time")
                
                if not combined_mask.is_empty():
                    decay_flag = True
                    decay_time = combined_mask.row(0)["date_time"]
                    
        except DataEmptyError:
            decay_flag, decay_time = False, ''
            o, h, l, c = '', '', '', ''
        except Exception as e:
            print('decay_check_single_leg', e)
            traceback.print_exc()
            decay_flag, decay_time = False, ''
            o, h, l, c = '', '', '', ''

        if with_ohlc:
            return o, h, l, c, decay_price, decay_flag, decay_time
        else:
            return decay_price, decay_flag, decay_time
        
    def __del__(self) -> None:
        print("Deleting instance ...")
