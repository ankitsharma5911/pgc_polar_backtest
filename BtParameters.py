import sys
import datetime
import argparse
import itertools
import pandas as pd

def get_dte_file(pickle_path):
    dte_file = pd.read_csv(f"{pickle_path}DTE.csv", parse_dates=['Date'], dayfirst=True).set_index("Date")
    return dte_file

def get_meta_data(code, meta_data_path):
    
    dtypes = {"index": "category", "dte":"Int8", "run":"bool"}
    meta_data = pd.read_csv(meta_data_path, dtype=dtypes)
    
    try:
        meta_data['from_date'] = pd.to_datetime(meta_data['from_date'], format="%d-%m-%Y")
    except:
        meta_data['from_date'] = pd.to_datetime(meta_data['from_date'], format="%Y-%m-%d")
    
    try:
        meta_data['to_date'] = pd.to_datetime(meta_data['to_date'], format="%d-%m-%Y")
    except:
        meta_data['to_date'] = pd.to_datetime(meta_data['to_date'], format="%Y-%m-%d")
    
    meta_data["start_time"] = pd.to_datetime(meta_data["start_time"], format="%H:%M:%S").dt.time
    meta_data["end_time"] = pd.to_datetime(meta_data["end_time"], format="%H:%M:%S").dt.time
        
    meta_row_nos = None
    if 'ipykernel' not in sys.modules:
        parser = argparse.ArgumentParser()
        parser.add_argument('-r', '--MetaRowNo', type=int)
        args = parser.parse_args()

        if args.MetaRowNo is not None:
            meta_row_nos = [args.MetaRowNo]
        
        title = f"{code} {meta_row_nos}"
            
        if sys.platform == 'linux':
            sys.stdout.write(f"\033]0;{title}\007")
            sys.stdout.flush()
        elif sys.platform == 'win32':
            import ctypes
            ctypes.windll.kernel32.SetConsoleTitleW(title)
        
    meta_row_nos = meta_row_nos or range(len(meta_data))
    return meta_data, meta_row_nos

def get_meta_row_data(meta_row, pickle_path, weekly=False):
    
    index, from_date, to_date, start_time, end_time = meta_row['index'], meta_row['from_date'], meta_row['to_date'], meta_row['start_time'], meta_row['end_time']
    dte_file = get_dte_file(pickle_path)

    if not weekly:
        dte = int(meta_row['dte'])
        date_lists = dte_file.loc[(dte_file.index >= from_date) & (dte_file.index <= to_date) & (dte_file[index] == dte)].index.to_list()
        return index, dte, from_date, to_date, start_time, end_time, date_lists
    
    elif weekly:
        from_dte, to_dte = int(meta_row['from_dte']), int(meta_row['to_dte'])
        date_lists = dte_file.loc[(dte_file.index >= from_date) & (dte_file.index <= to_date) & (~dte_file[index].isna())].index.to_list()
        
        week_dates, week_lists = [], []
        prev_dte = 99

        for date in date_lists:
            dte = int(dte_file.loc[date, index])

            if prev_dte > dte:
                week_dates.append(date)
            else:
                week_lists.append(week_dates)
                week_dates = [date]

            prev_dte = dte

        if week_dates:
            week_lists.append(week_dates)

        return index, from_dte, to_dte, from_date, to_date, start_time, end_time, week_lists

def get_parameter_data(code, parameter_path):
    
    parameter = pd.read_csv(parameter_path)
    for col in parameter.columns:
        if ('time' in col) and ('and' not in col):
            globals()[f'{col}'] = pd.to_datetime(parameter[col].dropna().str.replace(' ', '').str[0:5], format='%H:%M').dt.time.to_list()
        else:
            globals()[f'{col}'] = parameter[col].dropna().to_list()
            
    parameter = pd.DataFrame(list(itertools.product(*[globals()[f'{col}'] for col in parameter.columns])), columns=parameter.columns)
    
    # filter - entry < (exit_time - 5min)
    parameter = parameter[pd.to_datetime(parameter['entry_time'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['exit_time'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]

    if code.endswith('_PSL') and "last_trade_time_and_interval" in parameter.columns:
        parameter[['last_trade_time', 'trade_interval']] = parameter['last_trade_time_and_interval'].str.strip().str.split(',', expand=True)
        parameter['last_trade_time'] = pd.to_datetime(parameter['last_trade_time'], format='%H:%M:%S').dt.time

    if (code == 'B120') or (code == 'B120_TTC_RE') or (code == 'B120W') or (code == 'B120M') or (code == 'B120_SI'):
    
        parameter.loc[parameter['sl'] == 0, 'ut_sl'] = 0
        parameter.loc[parameter['sl'] == 0, 'method'] = 'HL'
        
        parameter['ut_sl'] = parameter['ut_sl'].astype(str).str.upper()
        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['method'] = parameter['method'].str.upper()
        
        if code == 'B120_SI':
            parameter['std_indicator'] = parameter['std_indicator'].str.upper()
 
    if (code == 'B120_RE_UT') or (code == "B120_UT_TRAIL"):
    
        parameter.loc[parameter['sl'] == 0, 'ut_sl'] = 0
        parameter.loc[parameter['sl'] == 0, 'method'] = 'HL'
        
        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['method'] = parameter['method'].str.upper()

    elif (code == 'B120_PSL') or (code == 'B120_SI_PSL') or (code == 'B120W_PSL'):
        
        # filter - entry < (exit_time|endtime - 5min)
        parameter = parameter[pd.to_datetime(parameter['entry_time'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['last_trade_time'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        parameter = parameter[pd.to_datetime(parameter['last_trade_time'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['exit_time'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        
        # filter - where sl = 0
        parameter.loc[parameter['sl'] == 0, 'ut_sl'] = 0
        parameter.loc[parameter['sl'] == 0, 'method'] = 'HL'
        
        parameter['ut_sl'] = parameter['ut_sl'].astype(str).str.upper()
        parameter['trade_interval'] = parameter['trade_interval'].str.upper()
        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['method'] = parameter['method'].str.upper()
        
        if code == 'B120_SI_PSL':
            parameter['std_indicator'] = parameter['std_indicator'].str.upper()
            

    if code == 'B120_UNIVERSAL':

        #filer intra sl
        parameter['intra_sl'] = parameter.apply(lambda row: row['sl'] + float(row['intra_sl'] .split('+')[-1]) if '+' in str(row['intra_sl']) else float(row['intra_sl']), axis=1)
        parameter = parameter[~((parameter['intra_sl'] != 0) & (parameter['intra_sl'] <= parameter['sl']))]
        
        parameter['ut_intra_sl'] = parameter.apply(lambda row: row['ut_sl'] + float(row['ut_intra_sl'] .split('+')[-1]) if '+' in str(row['ut_intra_sl']) else float(row['ut_intra_sl']), axis=1)
        parameter = parameter[~((parameter['ut_intra_sl'] != 0) & (parameter['ut_intra_sl'] <= parameter['ut_sl']))]

        # filter - where sl = 0
        parameter.loc[parameter['sl'] == 0, ['intra_sl', 'ut_sl', 'ut_intra_sl']] = 0
        parameter.loc[parameter['sl'] == 0, 'method'] = 'HL'
        parameter.loc[parameter['ut_sl'] == 0, 'ut_intra_sl'] = 0
        
        #filter where method HL and Intra SL
        parameter.loc[parameter['method'] == 'HL', 'intra_sl'] = 0
        parameter.loc[parameter['method'] == 'HL', 'ut_intra_sl'] = 0
        
        parameter['ut_sl'] = parameter['ut_sl'].astype(str).str.upper()
        parameter.loc[parameter['ut_sl'] == 'TTC', 'ut_intra_sl'] = 0
        
        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['method'] = parameter['method'].str.upper()
        
    
    elif (code == 'B120_RE') or (code == 'B120_TTC_RE') or (code == 'B120W_RE'):
        
        # filter - where sl = 0
        parameter.loc[parameter['sl'] == 0, 'ut_sl'] = 0
        parameter.loc[parameter['sl'] == 0, 'method'] = 'HL'
        
        parameter['ut_sl'] = parameter['ut_sl'].astype(str).str.upper()
        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['method'] = parameter['method'].str.upper()
        
        
    elif (code == 'B120_RE_PSL') or (code == 'B120_DUT_RE_PSL') or (code == 'B120_RE_SI_PSL') or (code == 'B120_TTC_RE_PSL') or (code == 'B120_TTC_RE_SI_PSL'):
            
        # filter - entry < (exit_time|endtime - 5min)
        parameter = parameter[pd.to_datetime(parameter['entry_time'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['last_trade_time'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        parameter = parameter[pd.to_datetime(parameter['last_trade_time'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['exit_time'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        
        # filter - where sl = 0
        parameter.loc[parameter['sl'] == 0, 'ut_sl'] = 0
        parameter.loc[parameter['sl'] == 0, 'method'] = 'HL'
        
        parameter['ut_sl'] = parameter['ut_sl'].astype(str).str.upper()
        parameter['trade_interval'] = parameter['trade_interval'].str.upper()
        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['method'] = parameter['method'].str.upper()
        
        if code == 'B120_RE_SI_PSL':
            parameter['std_indicator'] = parameter['std_indicator'].str.upper()
            
    elif (code == 'B120G'):
        
        parameter = parameter[pd.to_datetime(parameter['exit_time'], format='%H:%M:%S').dt.time <= (pd.to_datetime(parameter['entry_time2'], format='%H:%M:%S')).dt.time]
        parameter = parameter[pd.to_datetime(parameter['entry_time2'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['exit_time2'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        
        # filter - where sl = 0
        parameter.loc[parameter['sl'] == 0, 'ut_sl'] = 0
        parameter.loc[parameter['sl2'] == 0, 'ut_sl2'] = 0
        parameter.loc[(parameter['sl'] == 0) | (parameter['sl2'] == 0), 'method'] = 'HL'
        
        parameter['ut_sl'] = parameter['ut_sl'].astype(str).str.upper()
        parameter['ut_sl2'] = parameter['ut_sl2'].astype(str).str.upper()
        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['method'] = parameter['method'].str.upper()

    elif (code == 'B120G_PSL'):
        
        # filter - entry < (exit_time|endtime - 5min)
        parameter = parameter[pd.to_datetime(parameter['exit_time'], format='%H:%M:%S').dt.time <= (pd.to_datetime(parameter['entry_time2'], format='%H:%M:%S')).dt.time]
        parameter = parameter[pd.to_datetime(parameter['entry_time2'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['exit_time2'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        
        parameter = parameter[pd.to_datetime(parameter['entry_time'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['last_trade_time'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        parameter = parameter[pd.to_datetime(parameter['last_trade_time'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['exit_time'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        
        parameter = parameter[pd.to_datetime(parameter['entry_time2'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['last_trade_time2'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        parameter = parameter[pd.to_datetime(parameter['last_trade_time2'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['exit_time2'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        
        def check_B120G_PSL(row):
            today = datetime.datetime.today()
            start_dt = datetime.datetime.combine(today, row['entry_time'])
            start_dt2 = datetime.datetime.combine(today, row['entry_time2'])
            last_trade_dt = datetime.datetime.combine(today, row['last_trade_time'])
            last_trade_dt2 = datetime.datetime.combine(today, row['last_trade_time2'])
            time_range = pd.date_range(start_dt, last_trade_dt, freq=row['trade_interval'].lower())
            time_range2 = pd.date_range(start_dt2, last_trade_dt2, freq=row['trade_interval2'].lower())
            return len(time_range) == len(time_range2)

        parameter = parameter[parameter.apply(lambda x : check_B120G_PSL(x), axis=1)]
        
        # filter - where sl = 0
        parameter.loc[parameter['sl'] == 0, 'ut_sl'] = 0
        parameter.loc[parameter['sl2'] == 0, 'ut_sl2'] = 0
        parameter.loc[(parameter['sl'] == 0) | (parameter['sl2'] == 0), 'method'] = 'HL'

        parameter['ut_sl'] = parameter['ut_sl'].astype(str).str.upper()
        parameter['ut_sl2'] = parameter['ut_sl2'].astype(str).str.upper()
        parameter['trade_interval'] = parameter['trade_interval'].str.upper()
        parameter['trade_interval2'] = parameter['trade_interval2'].str.upper()
        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['method'] = parameter['method'].str.upper()


    elif (code == 'DT') or (code == 'DT_SI') or (code == 'DT_RE') or (code =='DT_TRAIL') or (code == 'DTN') or (code == 'DTB_RE') or (code == 'DTW'):
        # filter - where sl = 0
        parameter.loc[parameter['sl'] == 0, 'method'] = 'HL'
        
        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['method'] = parameter['method'].str.upper()
        
        if code == 'DT_SI':
            parameter['std_indicator'] = parameter['std_indicator'].str.upper()

        if code =='DT_TRAIL':
            parameter = parameter[parameter['trail_profit'] >= parameter['trail_sl']]
                        
            parameter.loc[parameter['sl'] == 0, 'trail_profit'] = 0
            parameter.loc[parameter['sl'] == 0, 'trail_sl'] = 0
            
            parameter.loc[parameter['trail_sl'] == 0, 'trail_profit'] = 0
            parameter.loc[parameter['trail_profit'] == 0, 'trail_sl'] = parameter['sl']


    elif (code == 'DT_FS_SRE'):
        # filter - where dt_sl = 0
        parameter.loc[parameter['dt_sl'] == 0, 'method'] = 'HL'
        
        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['method'] = parameter['method'].str.upper()
        
        #filer intra sl
        parameter['sre_intra_sl'] = parameter.apply(lambda row: row['sre_sl'] + float(row['sre_intra_sl'].split('+')[-1]) if '+' in str(row['sre_intra_sl']) else float(row['sre_intra_sl']), axis=1)
        parameter = parameter[~((parameter['sre_intra_sl'] != 0) & (parameter['sre_intra_sl'] <= parameter['sre_sl']))]
        
    elif (code == 'DT_FS_SUT'):
        # filter - where dt_sl = 0
        parameter.loc[parameter['dt_sl'] == 0, 'method'] = 'HL'
        
        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['method'] = parameter['method'].str.upper()
        
        #filer intra sl
        parameter['sut_intra_sl'] = parameter.apply(lambda row: row['sut_sl'] + float(row['sut_intra_sl'].split('+')[-1]) if '+' in str(row['sut_intra_sl']) else float(row['sut_intra_sl']), axis=1)
        parameter = parameter[~((parameter['sut_intra_sl'] != 0) & (parameter['sut_intra_sl'] < parameter['sut_sl']))]

    elif (code == 'DT_FS_UT'):
        # filter - where fsl = 0
        parameter.loc[(parameter['fsl'] == 0) & (parameter['ssl'] == 0), 'method'] = 'HL'
        
        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['method'] = parameter['method'].str.upper()
        
    elif (code == 'OPTIONS_RANGE_BREAKOUT'):
        
        parameter.loc[parameter['sl'] == 0, 'method'] = 'HL'
        
        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['method'] = parameter['method'].str.upper()
        
    elif (code == 'FUTURE_RANGE_BREAKOUT'):
        
        parameter.loc[parameter['sl'] == 0, 'method'] = 'HL'
                
        parameter['method'] = parameter['method'].str.upper()

    elif (code == 'DT_PSL') or (code == 'DT_SI_PSL') or (code == 'DT_RE_PSL'):
        parameter = parameter[pd.to_datetime(parameter['entry_time'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['last_trade_time'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        parameter = parameter[pd.to_datetime(parameter['last_trade_time'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['exit_time'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        
        # filter - where sl = 0
        parameter.loc[parameter['sl'] == 0, 'method'] = 'HL'
        
        parameter['trade_interval'] = parameter['trade_interval'].str.upper()
        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['method'] = parameter['method'].str.upper()
        
        if code == 'DT_SI_PSL':
            parameter['std_indicator'] = parameter['std_indicator'].str.upper()

            
    elif (code == 'NRE') or (code == 'NREW') or (code == 'NRE_SI') or (code == 'NRE_CC') or (code == 'NREW_CC'):
        # filter - where sl = 0
        parameter.loc[parameter['sl'] == 0, 'method'] = 'HL'
        parameter.loc[parameter['sl'] == 0, 're_sl'] = 0
        
        parameter['re_sl'] = parameter.apply(lambda row: row['sl'] + float(row['re_sl'].split('+')[-1]) if '+' in str(row['re_sl']) else float(row['re_sl']), axis=1)
        
        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['method'] = parameter['method'].str.upper()
        
        if code == 'NRE_SI':
            parameter['std_indicator'] = parameter['std_indicator'].str.upper()
            
    elif (code == 'NREW_PSL'):
        
        # filter - entry < (exit_time - 5min)
        parameter = parameter[pd.to_datetime(parameter['entry_time'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['last_trade_time'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        parameter = parameter[pd.to_datetime(parameter['last_trade_time'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['exit_time'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        
        # filter - where sl = 0
        parameter.loc[parameter['sl'] == 0, 'method'] = 'HL'
        
        parameter['re_sl'] = parameter.apply(lambda row: row['sl'] + float(row['re_sl'].split('+')[-1]) if '+' in str(row['re_sl']) else float(row['re_sl']), axis=1)
        
        parameter['trade_interval'] = parameter['trade_interval'].str.upper()
        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['method'] = parameter['method'].str.upper()

    elif (code == 'NRE_PSL') or (code == 'NRE_SI_PSL') or (code == 'NRE_CC_PSL'):
        
        # filter - entry < (exit_time - 5min)
        parameter = parameter[pd.to_datetime(parameter['entry_time'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['last_trade_time'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        parameter = parameter[pd.to_datetime(parameter['last_trade_time'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['exit_time'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        
        # filter - where sl = 0
        parameter.loc[parameter['sl'] == 0, 'method'] = 'HL'
        
        parameter['trade_interval'] = parameter['trade_interval'].str.upper()
        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['method'] = parameter['method'].str.upper()
        
        if code == 'NRE_SI_PSL':
            parameter['std_indicator'] = parameter['std_indicator'].str.upper()


    elif (code == 'RED') or (code == 'RED_SI') or (code == 'REDW'):
        # filter - where sl = 0
        parameter.loc[parameter['sl'] == 0, 'method'] = 'HL'
        
        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['method'] = parameter['method'].str.upper()
        
        if code == 'RED_SI':
            parameter['std_indicator'] = parameter['std_indicator'].str.upper()


    elif (code == 'RED_PSL') or (code == 'RED_SI_PSL'):
        
        parameter = parameter[pd.to_datetime(parameter['entry_time'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['last_trade_time'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        parameter = parameter[pd.to_datetime(parameter['last_trade_time'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['exit_time'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        
        # filter - where sl = 0
        parameter.loc[parameter['sl'] == 0, 'method'] = 'HL'
        
        parameter.loc[parameter['re_entries'] == 0, 'decay'] = parameter['decay'].min()
        
        parameter['trade_interval'] = parameter['trade_interval'].str.upper()
        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['method'] = parameter['method'].str.upper()
        
        if code == 'RED_SI_PSL':
            parameter['std_indicator'] = parameter['std_indicator'].str.upper()


    elif (code == 'SBS') or (code == 'SBS_SI'):
        
        # filter - where sl = 0
        parameter.loc[(parameter['sell_sl'] == 0), 'method'] = 'HL'
        parameter.loc[(parameter['sell_sl'] == 0), 'sell_trail'] = 0
        parameter.loc[(parameter['sell_sl'] == 0), 'sell_sl_trail'] = 0
        parameter.loc[(parameter['sell_sl'] == 0), 'sell_track_original'] = True
        parameter.loc[(parameter['sell_sl'] == 0), 'buy_flag'] = False        
        parameter.loc[(parameter['sell_sl'] == 0), 'buy_sl'] = 0
        parameter.loc[(parameter['sell_sl'] == 0), 'buy_trail'] = 0
        parameter.loc[(parameter['sell_sl'] == 0), 'buy_sl_trail'] = 0
        parameter.loc[(parameter['sell_sl'] == 0), 'buy_track_original'] = False
        parameter.loc[(parameter['sell_sl'] == 0), 'sell2_flag'] = False
        
        # parameter = parameter[parameter['sell_trail'] >= parameter['sell_sl_trail']]
        parameter = parameter[parameter['sell_trail'] == parameter['sell_sl_trail']]
        parameter.loc[parameter['sell_sl_trail'] == 0, 'sell_trail'] = 0
        parameter.loc[parameter['sell_trail'] == 0, 'sell_sl_trail'] = parameter['sell_sl']
    
        # parameter = parameter[parameter['buy_trail'] >= parameter['buy_sl_trail']]
        parameter = parameter[parameter['buy_trail'] == parameter['buy_sl_trail']]
        parameter.loc[parameter['buy_sl_trail'] == 0, 'buy_trail'] = 0
        parameter.loc[parameter['buy_trail'] == 0, 'buy_sl_trail'] = parameter['buy_sl']       
                        
        # where buy_flag is False
        parameter.loc[parameter['buy_flag'] == False, ['buy_sl','buy_sl_trail','buy_trail']] = 0
        parameter.loc[parameter['buy_flag'] == False, ['buy_track_original']] = False
        parameter.loc[parameter['buy_flag'] == False, ['sell2_flag']] = False
        
        # where buy_sl is 0
        parameter.loc[(parameter['buy_sl'] == 0), 'buy_trail'] = 0
        parameter.loc[(parameter['buy_sl'] == 0), 'buy_sl_trail'] = 0
        parameter.loc[(parameter['buy_sl'] == 0), 'buy_track_original'] = True
        parameter.loc[(parameter['buy_sl'] == 0), 'sell2_flag'] = False        

        parameter['method'] = parameter['method'].str.upper()
        
        if code == 'SBS_SI':
            parameter['std_indicator'] = parameter['std_indicator'].str.upper()
    
    elif (code == 'SBS_PSL') or (code == 'SBS_SI_PSL'):
        
        parameter = parameter[pd.to_datetime(parameter['entry_time'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['last_trade_time'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        parameter = parameter[pd.to_datetime(parameter['last_trade_time'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['exit_time'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        
        # filter - where sl = 0
        parameter.loc[(parameter['sell_sl'] == 0), 'method'] = 'HL'
        
        parameter.loc[(parameter['sell_sl'] == 0) & (parameter['sell_trail'] == 0), 'buy_sl'] = 0
        parameter.loc[(parameter['sell_sl'] == 0) & (parameter['sell_trail'] == 0), 'buy_trail'] = 0
        parameter.loc[(parameter['sell_sl'] == 0) & (parameter['sell_trail'] == 0), 'buy_sl_trail'] = 0
        
        parameter.loc[parameter['sell_sl'] == 0, 'sell_sl_trail'] = 0
        parameter.loc[parameter['buy_sl'] == 0, 'buy_sl_trail'] = 0
        
        # filter - where trail = 0
        parameter.loc[parameter['sell_trail'] == 0, 'sell_sl_trail'] = 0
        parameter.loc[parameter['buy_trail'] == 0, 'buy_sl_trail'] = 0
        parameter.loc[parameter['buy_flag'] == False, ['buy_sl','buy_sl_trail','buy_trail']] = 0
        parameter.loc[parameter['buy_flag'] == False, ['sell2_flag']] = False

        parameter['method'] = parameter['method'].str.upper()
        
        if code == 'SBS_SI_PSL':
            parameter['std_indicator'] = parameter['std_indicator'].str.upper()


    elif (code == 'SRE') or (code == 'SREW') or (code == 'SRE_SI') or (code == 'CSRE'):

        #filer intra sl
        parameter['intra_sl'] = parameter.apply(lambda row: row['sl'] + float(row['intra_sl'].split('+')[-1]) if '+' in str(row['intra_sl']) else float(row['intra_sl']), axis=1)
        parameter = parameter[~((parameter['intra_sl'] != 0) & (parameter['intra_sl'] <= parameter['sl']))]

        parameter['orderside'] = parameter['orderside'].str.upper()
        
        if code == 'SRE_SI':
            parameter['std_indicator'] = parameter['std_indicator'].str.upper()
            

    elif (code == 'SRE_SEPARATE_LEG_SL') or (code == 'SREW_SEPARATE_LEG_SL'):

        parameter['orderside'] = parameter['orderside'].str.upper()

    elif (code == 'CSRE_HCLP'):

        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['method'] = parameter['method'].str.upper()

    elif (code == 'SRE_PREMIUM_SHIFT') or (code == 'SRE_PREMIUM_SHIFT_PSL') or (code == 'SRE_PREMIUM_SHIFT_TRAIL') or (code == 'SRE_PREMIUM_SHIFT_TRAIL_PSL') or (code == 'SRE_PREMIUM_SHIFT_ACTION') or (code == 'SRE_PREMIUM_SHIFT_ACTION_PSL'):
        
        parameter['orderside'] = parameter['orderside'].str.upper()
        

    elif (code == 'SREW_RANGE') or (code == 'SREW_RANGE_TRAIL') or (code == 'SREW_RANGE_STR_TO_STD'):

        #filer intra sl
        parameter['intra_sl'] = parameter.apply(lambda row: row['sl'] + float(row['intra_sl'].split('+')[-1]) if '+' in str(row['intra_sl']) else float(row['intra_sl']), axis=1)
        parameter = parameter[~((parameter['intra_sl'] != 0) & (parameter['intra_sl'] <= parameter['sl']))]

        parameter['fixed_or_dynamic'] = parameter['fixed_or_dynamic'].str.upper()
        parameter['normal_or_cut'] = parameter['normal_or_cut'].str.upper()
        parameter['orderside'] = parameter['orderside'].str.upper()
        
    elif code == "SREW_RANGE_PSL":
        
        parameter = parameter[pd.to_datetime(parameter['entry_time'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['last_trade_time'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        parameter = parameter[pd.to_datetime(parameter['last_trade_time'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['exit_time'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        
        parameter['intra_sl'] = parameter.apply(lambda row: row['sl'] + float(row['intra_sl'].split('+')[-1]) if '+' in str(row['intra_sl']) else float(row['intra_sl']), axis=1)
        parameter = parameter[~((parameter['intra_sl'] != 0) & (parameter['intra_sl'] <= parameter['sl']))]

        parameter['fixed_or_dynamic'] = parameter['fixed_or_dynamic'].str.upper()
        parameter['normal_or_cut'] = parameter['normal_or_cut'].str.upper()
        parameter['orderside'] = parameter['orderside'].str.upper()


    elif (code == 'SRE_PSL') or (code == 'SRE_SI_PSL') or (code == 'SREW_PSL'):
        
        parameter = parameter[pd.to_datetime(parameter['entry_time'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['last_trade_time'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        parameter = parameter[pd.to_datetime(parameter['last_trade_time'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['exit_time'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        
        #filer intra sl
        parameter['intra_sl'] = parameter.apply(lambda row: row['sl'] + float(row['intra_sl'].split('+')[-1]) if '+' in str(row['intra_sl']) else float(row['intra_sl']), axis=1)
        parameter = parameter[~((parameter['intra_sl'] != 0) & (parameter['intra_sl'] < parameter['sl']))]

        parameter['trade_interval'] = parameter['trade_interval'].str.upper()
        parameter['orderside'] = parameter['orderside'].str.upper()
        
        if code == 'SRE_SI_PSL':
            parameter['std_indicator'] = parameter['std_indicator'].str.upper()
            
            
    elif (code == 'SRE_SEPARATE_LEG_SL_PSL'):
        
        parameter = parameter[pd.to_datetime(parameter['entry_time'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['last_trade_time'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        parameter = parameter[pd.to_datetime(parameter['last_trade_time'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['exit_time'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        
        parameter['trade_interval'] = parameter['trade_interval'].str.upper()
        parameter['orderside'] = parameter['orderside'].str.upper()
            
    
    elif (code == 'SUT') or (code == 'SUTW') or (code == 'SUT_SI'):

        #filer intra sl
        parameter['intra_sl'] = parameter.apply(lambda row: row['sl'] + float(row['intra_sl'].split('+')[-1]) if '+' in str(row['intra_sl']) else float(row['intra_sl']), axis=1)
        parameter = parameter[~((parameter['intra_sl'] != 0) & (parameter['intra_sl'] < parameter['sl']))]

        # filter - where sl = 0 & intra_sl = 0
        parameter.loc[(parameter['sl'] == 0) & (parameter['intra_sl'] == 0), 'ut_sl'] = 0

        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['ut_orderside'] = parameter['ut_orderside'].str.upper()
        parameter['ut_method'] = parameter['ut_method'].str.upper()
        
        if code == 'SUT_SI':
            parameter['std_indicator'] = parameter['std_indicator'].str.upper()
            
    elif (code == 'SUT_TT'):

        #filer intra sl
        parameter['intra_sl'] = parameter.apply(lambda row: row['sl'] + float(row['intra_sl'].split('+')[-1]) if '+' in str(row['intra_sl']) else float(row['intra_sl']), axis=1)
        parameter = parameter[~((parameter['intra_sl'] != 0) & (parameter['intra_sl'] < parameter['sl']))]

        # filter - where sl = 0 & intra_sl = 0
        parameter.loc[(parameter['sl'] == 0) & (parameter['intra_sl'] == 0), 'ut_sl'] = 0

        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['ut_orderside'] = parameter['ut_orderside'].str.upper()
        parameter['ut_method'] = parameter['ut_method'].str.upper()
        parameter['tt_orderside'] = parameter['tt_orderside'].str.upper()
        parameter['tt_method'] = parameter['ut_method'].str.upper()
        
        if code == 'SUT_SI':
            parameter['std_indicator'] = parameter['std_indicator'].str.upper()
    
    
    elif (code == 'SUT_PSL') or (code == 'SUT_SI_PSL') or (code == 'SUTW_PSL'):
        
        parameter = parameter[pd.to_datetime(parameter['entry_time'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['last_trade_time'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        parameter = parameter[pd.to_datetime(parameter['last_trade_time'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['exit_time'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        
        #filer intra sl
        parameter['intra_sl'] = parameter.apply(lambda row: row['sl'] + float(row['intra_sl'].split('+')[-1]) if '+' in str(row['intra_sl']) else float(row['intra_sl']), axis=1)
        parameter = parameter[~((parameter['intra_sl'] != 0) & (parameter['intra_sl'] < parameter['sl']))]

        # filter - where sl = 0 & intra_sl = 0
        parameter.loc[(parameter['sl'] == 0) & (parameter['intra_sl'] == 0), 'ut_sl'] = 0

        parameter['trade_interval'] = parameter['trade_interval'].str.upper()
        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['ut_orderside'] = parameter['ut_orderside'].str.upper()
        parameter['ut_method'] = parameter['ut_method'].str.upper()
        
        if code == 'SUT_SI_PSL':
            parameter['std_indicator'] = parameter['std_indicator'].str.upper()
        
    elif (code == 'SUT_TT_PSL'):
        
        parameter = parameter[pd.to_datetime(parameter['entry_time'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['last_trade_time'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        parameter = parameter[pd.to_datetime(parameter['last_trade_time'], format='%H:%M:%S').dt.time < (pd.to_datetime(parameter['exit_time'], format='%H:%M:%S')-pd.Timedelta(minutes=5)).dt.time]
        
        #filer intra sl
        parameter['intra_sl'] = parameter.apply(lambda row: row['sl'] + float(row['intra_sl'].split('+')[-1]) if '+' in str(row['intra_sl']) else float(row['intra_sl']), axis=1)
        parameter = parameter[~((parameter['intra_sl'] != 0) & (parameter['intra_sl'] < parameter['sl']))]

        # filter - where sl = 0 & intra_sl = 0
        parameter.loc[(parameter['sl'] == 0) & (parameter['intra_sl'] == 0), 'ut_sl'] = 0

        parameter['trade_interval'] = parameter['trade_interval'].str.upper()
        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['ut_orderside'] = parameter['ut_orderside'].str.upper()
        parameter['ut_method'] = parameter['ut_method'].str.upper()
        parameter['tt_orderside'] = parameter['tt_orderside'].str.upper()
        parameter['tt_method'] = parameter['tt_method'].str.upper()
        
    elif code == 'SUT_SRE':
        #filer intra sl
        parameter['intra_sl'] = parameter.apply(lambda row: row['sl'] + float(row['intra_sl'].split('+')[-1]) if '+' in str(row['intra_sl']) else float(row['intra_sl']), axis=1)
        parameter = parameter[~((parameter['intra_sl'] != 0) & (parameter['intra_sl'] < parameter['sl']))]

        parameter['sre_intra_sl'] = parameter.apply(lambda row: row['sre_sl'] + float(row['sre_intra_sl'] .split('+')[-1]) if '+' in str(row['sre_intra_sl']) else float(row['sre_intra_sl']), axis=1)
        parameter = parameter[~((parameter['sre_intra_sl'] != 0) & (parameter['sre_intra_sl'] < parameter['sre_sl']))]

        # filter - where sl = 0 & intra_sl = 0
        parameter.loc[(parameter['sl'] == 0) & (parameter['intra_sl'] == 0), 'ut_sl'] = 0
        parameter.loc[parameter['ut_sl'] == 0, 'sre_sl'] = 0
        parameter.loc[parameter['ut_sl'] == 0, 'sre_intra_sl'] = 0

        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['ut_orderside'] = parameter['ut_orderside'].str.upper()
        parameter['ut_method'] = parameter['ut_method'].str.upper()
        
    
    elif code == 'S2UT':
        
        #filer intra sl
        parameter['intra_sl'] = parameter.apply(lambda row: row['sl'] + float(row['intra_sl'] .split('+')[-1]) if '+' in str(row['intra_sl']) else float(row['intra_sl']), axis=1)
        parameter = parameter[~((parameter['intra_sl'] != 0) & (parameter['intra_sl'] < parameter['sl']))]

        # filter - where sl = 0 & intra_sl = 0
        parameter.loc[(parameter['sl'] == 0) & (parameter['intra_sl'] == 0), 'ut_sl'] = 0

        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['ut_orderside'] = parameter['ut_orderside'].str.upper()
        parameter['ut_method'] = parameter['ut_method'].str.upper()
        
    elif (code == 'SRE_PS_W_Straddle') or (code == 'SRE_PS_W_Straddle_PSL'):
        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['method'] = parameter['method'].str.upper()
        
        parameter = parameter[parameter['movement1'] >= 50]
        parameter = parameter[parameter['movement2'] >= 100]

    elif code == 'IRONFLY':
        parameter = parameter[parameter['sl'] != 0]
    
    elif code == 'StraddleSELL_LaterHedgeBRE':
        parameter['method'] = parameter['method'].str.upper()
    
    elif code == 'MAC':
        
        parameter = parameter[parameter['short_period'] < parameter['long_period']]
        parameter['orderside'] = parameter['orderside'].str.upper()
        parameter['short_type'] = parameter['short_type'].str.upper()
        parameter['long_type'] = parameter['long_type'].str.upper()

    parameter.drop_duplicates(inplace=True, ignore_index=True)
    return parameter, len(parameter)

