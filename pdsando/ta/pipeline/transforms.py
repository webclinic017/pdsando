import pytz
from pytz import timezone
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pdpipe import PdPipelineStage
from pdsando.ta.pipeline.filters import RemoveNonMarketHours
import mplfinance as mpf

class Transform(PdPipelineStage):
  
  def __init__(self, **kwargs):
    super().__init__(exmsg='Transform failure', desc='Transform')
  
  def _prec(self, df):
    return True

class ColKeep(Transform):
  
  def __init__(self, columns, **kwargs):
    self._columns = columns
    super().__init__()
  
  def _transform(self, df, verbose):
    if verbose:
      print('Keeping only the following columns: "{}"'.format(','.join(self._columns)))
    return df[self._columns]

class Shift(Transform):
  
  def __init__(self, tgt_col, src_col, shift, **kwargs):
    self._src_col = src_col
    self._tgt_col = tgt_col
    self._shift = shift
    super().__init__()
  
  def _transform(self, df, verbose):
    ret_df = df.copy()
    if verbose:
      print('Adding Shifted col "{}"'.format(self._tgt_col))
    ret_df[self._tgt_col] = ret_df[self._src_col].shift(self._shift)
    return ret_df

class ToDateTime(Transform):
  
  def __init__(self, tgt_col, src_col, unit='ms', to_tz='utc', **kwargs):
    self._tgt_col = tgt_col
    self._src_col = src_col
    self._unit = unit
    self._to_tz = to_tz
    super().__init__()
  
  def _transform(self, df, verbose):
    ret_df = df.copy()
    
    if verbose:
      print('Converting epoch timestamp to human readable timestamp for "{}"'.format(self._src_col))
    
    ret_df[self._tgt_col] = pd.to_datetime(ret_df[self._src_col], utc=True, unit=self._unit)
    
    if self._to_tz != 'utc':
      ret_df[self._tgt_col] = ret_df[self._tgt_col].dt.tz_convert(self._to_tz)
    
    return ret_df

class ResetIndex(Transform):
  
  def __init__(self, drop=True, **kwargs):
    self._drop = drop
    super().__init__()
  
  def _transform(self, df, verbose):
    ret_df = df.copy()
    
    if verbose:
      print('Resetting DataFrame index')
    
    return ret_df.reset_index(drop=self._drop)

class MinVal(Transform):
  
  def __init__(self, tgt_col, col_list, **kwargs):
    self._tgt_col = tgt_col
    self._col_list = col_list
    super().__init__()
  
  def _transform(self, df, verbose):
    ret_df = df.copy()
    
    if verbose:
      print('Determining minimum value contained in columns: "{}"'.format(self._col_list))
    
    ret_df[self._tgt_col] = ret_df[self._col_list].min(axis=1)
    
    return ret_df

class MaxVal(Transform):
  
  def __init__(self, tgt_col, col_list, **kwargs):
    self._tgt_col = tgt_col
    self._col_list = col_list
    super().__init__()
  
  def _transform(self, df, verbose):
    ret_df = df.copy()
    
    if verbose:
      print('Determining minimum value contained in columns: "{}"'.format(self._col_list))
    
    ret_df[self._tgt_col] = ret_df[self._col_list].max(axis=1)
    
    return ret_df

class FillMissingTimeFrames(Transform):
  
  def __init__(self, delta, timestamp='Timestamp', from_time='9:30', to_time='16:00', **kwargs):
    self._delta = delta
    self._timestamp = timestamp
    self._from_time = from_time
    self._to_time = to_time
    super().__init__()
  
  def _per_delta(self, delta=1, start_hour=9, start_minute=30, end_hour=16, end_minute=0):
    time_of_day = []
    
    start = start_hour*60 + start_minute
    end = end_hour*60 + end_minute
    cur = start
    
    while cur < end:
      time_of_day.append(cur)
      cur += delta
      
    return pd.DataFrame({ 'time_of_day' : time_of_day }).set_index('time_of_day')
  
  def _transform(self, df, verbose):
    if verbose:
      print('Filling missing timeframes with delta "{}" between {} and {}'.format(
        self._delta,
        self._from_time,
        self._to_time
      ))
    
    dt_ref = self._per_delta(
      delta = self._delta,
      start_hour = int(self._from_time.split(':')[0]),
      start_minute = int(self._from_time.split(':')[1]),
      end_hour = int(self._to_time.split(':')[0]),
      end_minute = int(self._to_time.split(':')[1])
    )
    
    j = dt_ref.join(df.set_index(df[self._timestamp].dt.hour*60 + df[self._timestamp].dt.minute), how='left').sort_values(by=self._timestamp).reset_index(drop=True)
    j.fillna(method='ffill')
    
    return j[list(df.columns)]

# TODO PERFORMANCE OPTIMIZATIONS
class ThirtyToSixty(Transform):
  
  def __init__(self, open='Open', high='High', low='Low', close='Cloe', volume='Volume', timestamp='Timestamp', prefilter_market_hours=True, **kwargs):
    self._open = open
    self._high = high
    self._low = low
    self._close = close
    self._volume = volume
    self._timestamp = timestamp
    self._prefilter_market_hours = prefilter_market_hours
    super().__init__()
  
  def _transform(self, df, verbose):
    ret_df = df.copy()
    
    if verbose:
      print('Reducing 30 minute candles to 60')
    
    if self._prefilter_market_hours:
      ret_df = RemoveNonMarketHours(self._timestamp).apply(ret_df)
    
    ret_df['_group'] = ret_df.apply(lambda row: '{} {}:{}'.format(
      row[self._timestamp].date() if row[self._timestamp].hour != 0 else row[self._timestamp].date()-timedelta(days=1),
      row[self._timestamp].hour if row[self._timestamp].minute != 0 else (row[self._timestamp] - timedelta(hours=1)).hour,
      30
    ), axis=1)
    
    cols_to_select = [
      self._open, self._high,
      self._low, self._close,
      self._volume, self._timestamp,
      '_group'
    ]
    
    ret_df = ret_df[cols_to_select].groupby('_group').agg({
      self._close: 'last',
      self._high: 'max',
      self._low: 'min',
      self._open: 'first',
      self._timestamp: 'min',
      self._volume: 'sum'
    }).sort_values(by=self._timestamp).reset_index(drop=True)
    
    return ret_df

class SetIndex(Transform):
  
  def __init__(self, new_index, **kwargs):
    self._new_index = new_index
    super().__init__()
  
  def _transform(self, df, verbose):
    ret_df = df.copy()
    
    if verbose:
      print('Setting new index to: "{}"'.format(self._new_index))
    
    return ret_df.set_index(self._new_index)

class Slice(Transform):
  
  def __init__(self, start=None, end=None, **kwargs):
    self._start = start
    self._end = end
    super().__init__()
  
  def _prec(self, df):
    if not self._start and not self._end:
      return False
    return True
  
  def _transform(self, df, verbose):
    ret_df = df.copy()
    
    if verbose:
      print('Returning slice from {} to {}"'.format(self._start, self._end))
    
    if self._start and self._end:
      return ret_df[self._start: self._end]
    elif self._start:
      return ret_df[self._start:]
    else:
      return ret_df[:self._end]

class IntradayGroups(Transform):
  
  def __init__(self, group_size=2, open='Open', high='High', low='Low', close='Close', volume='Volume', timestamp='Timestamp', **kwargs):
    self._group_size = group_size
    self._open = open
    self._high = high
    self._low = low
    self._close = close
    self._volume = volume
    self._timestamp = timestamp
    super().__init__()
  
  def _transform(self, df, verbose):
    ret_df = df.copy()
    
    if verbose:
      print('Grouping every {} records within each day"'.format(self._group_size))
    
    ret_df['_date_group'] = ret_df[self._timestamp].dt.date
    g = ret_df.groupby(ret_df['_date_group']).cumcount() // self._group_size
    
    return ret_df.groupby(['_date_group', g]).agg({
      self._close: 'last',
      self._high: 'max',
      self._low: 'min',
      self._open: 'first',
      self._timestamp: 'min',
      self._volume: 'sum'
    }).sort_values(by=self._timestamp).reset_index(drop=True)

class BuySellOld(Transform):
  
  def __init__(self, tgt_col, src_col, short=False, **kwargs):
    self._tgt_col = tgt_col
    self._src_col = src_col
    self._short = short
    super().__init__()
  
  def _transform(self, df, verbose):
    ret_df = df.copy()
    
    if verbose:
      print('Converting raw signals ({}) to BuySell timeline events ({})"'.format(self._src_col, self._tgt_col))
    
    in_pos = False if not self._short else True
    ret_df[self._tgt_col] = np.nan
    for i in range(len(ret_df)):
      if ret_df[self._src_col].iat[i] > 0 and not in_pos:
        ret_df[self._tgt_col].iat[i] = 1
        in_pos = True
      elif ret_df[self._src_col].iat[i] < 0 and in_pos:
        ret_df[self._tgt_col].iat[i] = -1
        in_pos = False
      elif (not self._short and in_pos) or (self._short and not in_pos):
        ret_df[self._tgt_col].iat[i] = 0
    
    return ret_df

class BuySell(Transform):
  
  def __init__(self, tgt_col, src_col, close='Close', high='High', trail_frac=None, **kwargs):
    self._tgt_col = tgt_col
    self._src_col = src_col
    self._close = close
    self._high = high
    self._trail_frac = trail_frac
    super().__init__()
  
  def _transform(self, df, verbose):
    ret_df = df.copy()
    
    if verbose:
      print('Converting raw signals ({}) to BuySell timeline events ({}) with trailing stop ({})"'.format(self._src_col, self._tgt_col, self._trail_frac))
    
    in_pos = False
    cur_stop_price = -1.0
    src_val = df[self._src_col]
    ret_df[self._tgt_col] = np.nan
    
    for i in range(len(ret_df)):
      if src_val.iat[i] > 0 and not in_pos:
        in_pos = True
        ret_df[self._tgt_col].iat[i] = 1
      elif in_pos:
        if (ret_df[self._close].iat[i] <= cur_stop_price) or (ret_df[self._close].iat[i] < 0):
          in_pos = False
          cur_stop_price = -1.0
          ret_df[self._tgt_col].iat[i] = -1
        else:
          cur_stop_price = max(cur_stop_price, ret_df[self._high].iat[i-1] * (1.0 - self._trail_frac)) if self._trail_frac else -1.0
          ret_df[self._tgt_col].iat[i] = 0
    
    return ret_df
