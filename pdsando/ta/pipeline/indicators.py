import pdpipe as pdp
import numpy as np
import pandas as pd
import mplfinance as mpf
from pdpipe import PdPipelineStage

class Indicator(PdPipelineStage):
  
  def __init__(self, **kwargs):
    self._tgt_col = kwargs.pop('tgt_col')
    self._color = kwargs.pop('color', 'black')
    self._width = kwargs.pop('width', 1)
    self._alpha = kwargs.pop('alpha', 1)
    self._panel = kwargs.pop('panel', 0)
    super().__init__(exmsg='Indicator failure', desc='Indicator')
  
  def _prec(self, df):
    return True
  
  def _get_or_apply(self, df):
    if self._tgt_col in df.columns:
      return df
    else:
      return self._transform(df, False)
  
  def _indicator(self, df):
    return [mpf.make_addplot(self._get_or_apply(df)[self._tgt_col], panel=self._panel, color=self._color, type='line', width=self._width, alpha=self._alpha)]

class SMA(Indicator):
  
  def __init__(self, tgt_col, src_col, period=5, **kwargs):
    self._tgt_col = tgt_col
    self._src_col = src_col
    self._period = period
    super().__init__(tgt_col=tgt_col, **kwargs)
  
  def _transform(self, df, verbose):
    ret_df = df.copy()
    if verbose:
      print('Determining Simple Moving Average for: "{}"'.format(self._src_col))
    ret_df[self._tgt_col] = ret_df[self._src_col].rolling(self._period).mean()
    return ret_df

class EMA(Indicator):
  
  def __init__(self, tgt_col, src_col, period=5, **kwargs):
    self._src_col = src_col
    self._tgt_col = tgt_col
    self._period = period
    super().__init__(tgt_col=tgt_col, **kwargs)
  
  def _transform(self, df, verbose):
    ret_df = df.copy()
    if verbose:
      print('Determining Exponential Moving Average (period={}) for: "{}"'.format(self._period, self._tgt_col))
    ret_df[self._tgt_col] = ret_df[self._src_col].ewm(span=self._period, min_periods=self._period, adjust=False, ignore_na=False).mean()
    return ret_df

class SMMA(Indicator):
  
  def __init__(self, tgt_col, src_col, period=5, **kwargs):
    self._src_col = src_col
    self._tgt_col = tgt_col
    self._period = period
    super().__init__(tgt_col=tgt_col, **kwargs)
  
  def _transform(self, df, verbose):
    ret_df = df.copy()
    if verbose:
      print('Determining Smoothed Moving Average (period={}) for: "{}"'.format(self._period, self._tgt_col))
    ret_df[self._tgt_col] = ret_df[self._src_col].ewm(alpha=1.0/self._period).mean().values.flatten()
    return ret_df

class RollingMax(Indicator):
  
  def __init__(self, tgt_col, src_col, period=5, **kwargs):
    self._tgt_col = tgt_col
    self._src_col = src_col
    self._period = period
    super().__init__(tgt_col=tgt_col, **kwargs)
  
  def _transform(self, df, verbose):
    ret_df = df.copy()
    if verbose:
      print('Determining Rolling Maximum (period={}) for: "{}"'.format(self._period, self._src_col))
    ret_df[self._tgt_col] = ret_df[self._src_col].rolling(self._period).max()
    return ret_df

class RateOfChange(Indicator):
  
  def __init__(self, tgt_col, src_col, period=5, **kwargs):
    self._tgt_col = tgt_col
    self._src_col = src_col
    self._period = period
    super().__init__(tgt_col=tgt_col, **kwargs)
  
  def _transform(self, df, verbose):
    ret_df = df.copy()
    
    if verbose:
      print('Determining Rate of Change (period={}) for: "{}"'.format(self._period, self._src_col))
    
    ret_df['_historical_val'] = ret_df[self._src_col].shift(self._period)
    ret_df[self._tgt_col] = ( (ret_df[self._src_col] - ret_df['_historical_val']) / ret_df['_historical_val'] ) * 100
    ret_df.drop('_historical_val', axis=1, inplace=True)
    
    return ret_df

# max(high - low, abs(high - close[1]), abs(low - close[1]))
class TrueRange(Indicator):
  
  def __init__(self, tgt_col, high='High', low='Low', close='Close', **kwargs):
    self._tgt_col = tgt_col
    self._high = high
    self._low = low
    self._close = close
    super().__init__(tgt_col=tgt_col, **kwargs)
  
  def _transform(self, df, verbose):
    ret_df = df.copy()
    ret_df['_last_close_'] = ret_df[self._close].shift(1)
    
    if verbose:
      print('Calculating True Range col "{}"'.format(self._tgt_col))
    
    ret_df['_tr_comp_a_'] = ret_df[self._high] - ret_df[self._low]
    ret_df['_tr_comp_b_'] = ret_df[self._high] - ret_df['_last_close_']
    ret_df['_tr_comp_c_'] = ret_df[self._low] - ret_df['_last_close_']
    ret_df[self._tgt_col] = ret_df[['_tr_comp_a_', '_tr_comp_b_', '_tr_comp_c_']].max(axis=1)
    
    ret_df.drop(['_last_close_', '_tr_comp_a_', '_tr_comp_b_', '_tr_comp_c_'], axis=1, inplace=True)
    
    return ret_df

class ATR(Indicator):
  
  def __init__(self, tgt_col, period=5, high='High', low='Low', close='Close', **kwargs):
    self._tgt_col = tgt_col
    self._high = high
    self._low = low
    self._close = close
    self._period = period
    super().__init__(tgt_col=tgt_col, **kwargs)
  
  def _transform(self, df, verbose):
    ret_df = df.copy()
    if verbose:
      print('Calculating True Range col "{}"'.format(self._tgt_col))
    
    pipeline = pdp.PdPipeline([
      TrueRange(self._tgt_col, self._high, self._low, self._close),
      SMMA(self._tgt_col, self._tgt_col, self._period)
    ])
    
    return pipeline.apply(ret_df)

class HL2(Indicator):
  
  def __init__(self, tgt_col, high='High', low='Low', **kwargs):
    self._tgt_col = tgt_col
    self._high = high
    self._low = low
    super().__init__(tgt_col=tgt_col, **kwargs)
  
  def _transform(self, df, verbose):
    ret_df = df.copy()
    
    if verbose:
      print('Calculating average between High and Low')
    
    ret_df[self._tgt_col] = (ret_df[self._high] + ret_df[self._low]) /2
    return ret_df

class SuperTrend(Indicator):
  
  def __init__(self, tgt_col, period=10, multiplier=3, high='High', low='Low', close='Close', as_offset=False, **kwargs):
    self._tgt_col = tgt_col
    self._period = period
    self._multiplier = multiplier
    self._high = high
    self._low = low
    self._close = close
    self._as_offset = as_offset
    super().__init__(tgt_col=tgt_col, **kwargs)
  
  def _transform(self, df, verbose):
    ret_df = df.copy()
    
    if verbose:
      print('Calculating final supertrend bands')
    
    ret_df = pdp.PdPipeline([
      HL2('_hl2', high=self._high, low=self._low),
      TrueRange('_tr', high=self._high, low=self._low, close=self._close),
      EMA('_atr', '_tr', period=self._period)
    ]).apply(ret_df)
    
    ret_df['_basic_lower_band'] = ret_df['_hl2']-(self._multiplier * ret_df['_atr'])
    ret_df['_basic_upper_band'] = ret_df['_hl2']+(self._multiplier * ret_df['_atr'])
    
    ret_df['_lower_band'] = 0.0
    ret_df['_upper_band'] = 0.0
    
    for i in range(self._period, len(ret_df)):
      ret_df['_lower_band'].iat[i] = max(ret_df['_basic_lower_band'].iat[i], ret_df['_lower_band'].iat[i-1]) if ret_df[self._close].iat[i-1] > ret_df['_lower_band'].iat[i-1] else ret_df['_basic_lower_band'].iat[i]
      ret_df['_upper_band'].iat[i] = min(ret_df['_basic_upper_band'].iat[i], ret_df['_upper_band'].iat[i-1]) if ret_df[self._close].iat[i-1] < ret_df['_upper_band'].iat[i-1] else ret_df['_basic_upper_band'].iat[i]
    
    ret_df['_trend'] = 1
    for i in range(self._period, len(ret_df)):
      if (ret_df['_trend'].iat[i-1] < 0 and ret_df[self._close].iat[i] > ret_df['_upper_band'].iat[i-1]):
        ret_df['_trend'].iat[i] = 1
      elif (ret_df['_trend'].iat[i-1] > 0 and ret_df[self._close].iat[i] < ret_df['_lower_band'].iat[i-1]):
        ret_df['_trend'].iat[i] = -1
      else:
        ret_df['_trend'].iat[i] = ret_df['_trend'].iat[i-1]
    
    if self._as_offset:
      ret_df[self._tgt_col] = ret_df.apply(lambda row: row['_hl2']-row['_lower_band'] if row['_trend'] > 0 else row['_hl2']-row['_upper_band'], axis=1)
    else:
      ret_df[self._tgt_col] = ret_df.apply(lambda row: row['_lower_band'] if row['_trend'] > 0 else row['_upper_band'], axis=1)
    
    ret_df.drop(['_hl2', '_tr', '_atr', '_basic_lower_band', '_basic_upper_band', '_lower_band', '_upper_band', '_trend'], axis=1, inplace=True)
    return ret_df
  
  def _indicator(self, df):
    orig = self._as_offset
    if not orig:
      self._as_offset = True
      st = self._transform(df, False)[self._tgt_col]
      self._as_offset = orig
    else:
      st = self._get_or_apply(df)[self._tgt_col]
    hl2 = HL2('hl2', high=self._high, low=self._low).apply(df)['hl2']
    ss_upper = np.where(st > 0, hl2 - st, np.nan)
    ss_lower = np.where(st < 0, hl2 - st, np.nan)
    ss_upper[:self._period] = np.nan
    ss_lower[:self._period] = np.nan
    
    return [
      mpf.make_addplot(ss_upper, panel=0, color='g', type='line', width=3, alpha=0.5),
      mpf.make_addplot(ss_lower, panel=0, color='r', type='line', width=3, alpha=0.5)
    ]

class DonchianRibbon(Indicator):
  
  def __init__(self, tgt_col, period=20, high='High', low='Low', close='Close', debug=False, **kwargs):
    self._tgt_col = tgt_col
    self._period = period
    self._high = high
    self._low = low
    self._close = close
    self._debug = debug
    
    if period < 10:
      raise ValueError('Period must be 10 or higher.')
    
    super().__init__(tgt_col=tgt_col, **kwargs)
  
  def _calc_trend(self, df, p, compare_to_main):
    ret_df = df
    
    ret_df['_hh'] = ret_df[self._high].rolling(p).max().shift(1)
    ret_df['_ll'] = ret_df[self._low].rolling(p).min().shift(1)
    
    ret_df['_trend'] = np.nan
    ret_df.loc[ret_df[self._close] > ret_df['_hh'], '_trend'] = 1
    ret_df.loc[ret_df[self._close] < ret_df['_ll'], '_trend'] = -1
    ret_df['_trend'] = ret_df['_trend'].fillna(method='ffill')
    
    ret_df['_final_trend'] = 0
    if compare_to_main:
      ret_df.loc[(ret_df['_trend'] > 0) & (ret_df['_main'] > 0), '_final_trend'] = 1
      ret_df.loc[(ret_df['_trend'] < 0) & (ret_df['_main'] < 0), '_final_trend'] = -1
    else:
      ret_df['_final_trend'] = ret_df['_trend']
    
    trend = ret_df['_final_trend']
    ret_df.drop(['_hh', '_ll', '_trend', '_final_trend'], inplace=True, axis=1)
    return trend
  
  def _transform(self, df, verbose):
    ret_df = df.copy()
    
    if verbose:
      print('Calculating Donchian channels for a period of: {}'.format(self._period))
    
    ret_df['_main'] = self._calc_trend(ret_df, self._period, False)
    ret_df['_t1'] = self._calc_trend(ret_df, self._period-1, True)
    ret_df['_t2'] = self._calc_trend(ret_df, self._period-2, True)
    ret_df['_t3'] = self._calc_trend(ret_df, self._period-3, True)
    ret_df['_t4'] = self._calc_trend(ret_df, self._period-4, True)
    ret_df['_t5'] = self._calc_trend(ret_df, self._period-5, True)
    ret_df['_t6'] = self._calc_trend(ret_df, self._period-6, True)
    ret_df['_t7'] = self._calc_trend(ret_df, self._period-7, True)
    ret_df['_t8'] = self._calc_trend(ret_df, self._period-8, True)
    ret_df['_t9'] = self._calc_trend(ret_df, self._period-9, True)
    
    ret_df[self._tgt_col] = ret_df[['_main', '_t1', '_t2', '_t3', '_t4', '_t5', '_t6', '_t7', '_t8', '_t9']].sum(axis=1).astype('int')
    
    if not self._debug:
      ret_df.drop(['_main', '_t1', '_t2', '_t3', '_t4', '_t5', '_t6', '_t7', '_t8', '_t9'], inplace=True, axis=1)
    
    return ret_df
  
  def _indicator(self, df):
    tmp = self._get_or_apply(df)[self._tgt_col]
    pos = np.where(tmp > 0, tmp, 0)
    neg = np.where(tmp <= 0, tmp.abs(), 0)
    return [
      mpf.make_addplot(pos, panel=1, color='g', type='bar', width=0.75, alpha=self._alpha),
      mpf.make_addplot(neg, panel=1, color='r', type='bar', width=0.75, alpha=self._alpha)
    ]

class Highest(Indicator):
  
  def __init__(self, tgt_col, src_col='High', period=5, shift=0, **kwargs):
    self._tgt_col = tgt_col
    self._src_col = src_col
    self._period = period
    self._shift = shift
    super().__init__(tgt_col=tgt_col, **kwargs)
  
  def _transform(self, df, verbose):
    ret_df = df.copy()
    
    if verbose:
      print('Calculating highest value for column "{}" over period {}'.format(self._src_col, self._period))
    
    ret_df[self._tgt_col] = ret_df[self._src_col].rolling(self._period).max().shift(self._shift)
    return ret_df

class Lowest(Indicator):
  
  def __init__(self, tgt_col, src_col='Low', period=5, shift=0, **kwargs):
    self._tgt_col = tgt_col
    self._src_col = src_col
    self._period = period
    self._shift = shift
    super().__init__(tgt_col=tgt_col, **kwargs)
  
  def _transform(self, df, verbose):
    ret_df = df.copy()
    
    if verbose:
      print('Calculating lowest value for column "{}" over period {}'.format(self._src_col, self._period))
    
    ret_df[self._tgt_col] = ret_df[self._src_col].rolling(self._period).min().shift(self._shift)
    return ret_df

class AverageDirectionalIndex(Indicator):
  
  def __init__(self, tgt_col, period=5, close='Close', high='High', low='Low', **kwargs):
    self._tgt_col = tgt_col
    self._close = close
    self._high = high
    self._low = low
    self._period = period
    super().__init__(tgt_col=tgt_col, **kwargs)
  
  def _transform(self, df, verbose):
    ret_df = df.copy()
    temp = pd.DataFrame(ret_df[[self._close, self._high, self._low]].copy())
    
    if verbose:
      print('Calculating average directional index over period {}'.format(self._period))
    
    pipeline = pdp.PdPipeline([
      pdp.ColByFrameFunc('up', lambda df: df[self._high] - df[self._high].shift(1)),
      pdp.ColByFrameFunc('down', lambda df: df[self._low].shift(1) - df[self._low]),
      pdp.ColByFrameFunc('pdm', lambda df: np.where(df['up'] > df['down'], df['up'], 0)),
      pdp.ColByFrameFunc('ndm', lambda df: np.where(df['down'] > df['up'], df['down'], 0)),
      SMMA('smma_pdm', 'pdm', self._period),
      SMMA('smma_ndm', 'ndm', self._period),
      ATR('atr', self._period, close=self._close),
      pdp.ColByFrameFunc('pdi', lambda df: (100 * df['smma_pdm'] / df['atr'])),
      pdp.ColByFrameFunc('ndi', lambda df: (100 * df['smma_ndm'] / df['atr'])),
      pdp.ColByFrameFunc('avg_di', lambda df: (
        ((df['pdi'] - df['ndi']) / (df['pdi'] + df['ndi'])).abs()
      )),
      SMMA('pre_adx', 'avg_di', self._period),
      pdp.ColByFrameFunc('adx', lambda df: (100 * df['pre_adx']))
    ])
    
    ret_df[self._tgt_col] = pipeline.apply(temp)['adx']
    
    return ret_df

class DeviationSpread(Indicator):
  
  def __init__(self, tgt_col, src_col='Close', period=100, **kwargs):
    self._tgt_col = tgt_col
    self._src_col = src_col
    self._period = period
    super().__init__(tgt_col=tgt_col, **kwargs)
  
  def _transform(self, df, verbose):
    ret_df = df.copy()
    
    if verbose:
      print('Determining Standard Deviation for: "{}"'.format(self._src_col))
    
    stddev = ret_df[self._src_col].rolling(self._period).std()
    median = ret_df[self._src_col].rolling(self._period).median()
    
    ret_df['{}_lowest'.format(self._tgt_col)]  = median - 2*stddev
    ret_df['{}_low'.format(self._tgt_col)]     = median - stddev
    ret_df['{}_mid'.format(self._tgt_col)]     = median
    ret_df['{}_high'.format(self._tgt_col)]    = median + stddev
    ret_df['{}_highest'.format(self._tgt_col)] = median + 2*stddev
    
    return ret_df
  
  def _indicator(self, df):
    tmp = self._get_or_apply(df)
    return [
      mpf.make_addplot(tmp['{}_lowest'.format(self._tgt_col)] , panel=0, color='black' , type='line', width=self._width, alpha=0.4),
      mpf.make_addplot(tmp['{}_low'.format(self._tgt_col)]    , panel=0, color='blue'  , type='line', width=self._width, alpha=0.4),
      mpf.make_addplot(tmp['{}_mid'.format(self._tgt_col)]    , panel=0, color='purple', type='line', width=self._width, alpha=0.4),
      mpf.make_addplot(tmp['{}_high'.format(self._tgt_col)]   , panel=0, color='blue'  , type='line', width=self._width, alpha=0.4),
      mpf.make_addplot(tmp['{}_highest'.format(self._tgt_col)], panel=0, color='black' , type='line', width=self._width, alpha=0.4)
    ]

class Backtest(Indicator):
  
  def __init__(self, tgt_col, signal_col, price_col='Close', start_amount=1000000, as_percent=False, **kwargs):
    self._tgt_col = tgt_col
    self._signal_col = signal_col
    self._price_col = price_col
    self._start_amount = start_amount
    self._as_percent = as_percent
    super().__init__(tgt_col=tgt_col, **kwargs)
  
  def _transform(self, df, verbose):
    ret_df = df.copy()
    
    if verbose:
      print('Backtesting via Buy/Sell signals from {} with starting amount {}'.format(self._signal_col, self._start_amount))
    
    cur_val = float(self._start_amount)
    free_val = cur_val
    shares_held = 0
    
    ret_df[self._tgt_col] = 0.0
    for i in range(len(ret_df)):
      if ret_df[self._signal_col].iat[i] == 1.0:
        ret_df[self._tgt_col].iat[i] = free_val
        shares_held = free_val // ret_df[self._price_col].iat[i]
        free_val -= (shares_held * ret_df[self._price_col].iat[i])
      elif ret_df[self._signal_col].iat[i] == -1.0:
        free_val += (shares_held * ret_df[self._price_col].iat[i])
        shares_held = 0
        ret_df[self._tgt_col].iat[i] = free_val
      else:
        ret_df[self._tgt_col].iat[i] = free_val + (shares_held * ret_df[self._price_col].iat[i])
    
    if self._as_percent:
      ret_df[self._tgt_col] = (ret_df[self._tgt_col] / self._start_amount) * 100 - 100.00
    
    return ret_df
  
  def _indicator(self, df):
    return [mpf.make_addplot(self._get_or_apply(df)[self._tgt_col], panel=2, color=self._color, type='line', width=self._width, alpha=self._alpha)]