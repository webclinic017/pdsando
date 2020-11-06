import pandas as pd
import numpy as np
import pdpipe as pdp
import mplfinance as mpf
from datetime import datetime, timedelta
from pdpipe import PdPipelineStage

from pdsando.ta.datafeeds.polygon import Polygon
from pdsando.ta.pipeline.indicators import SuperTrend, DonchianRibbon, EMA, RollingMax, RateOfChange, HL2, AverageDirectionalIndex, BuySell
from pdsando.ta.pipeline.filters import RemoveNonMarketHours
from pdsando.ta.pipeline.transforms import Shift, ResetIndex, ThirtyToSixty, ColKeep, IntradayGroups

class Strategy(PdPipelineStage):
  
  def __init__(self, **kwargs):
    self._tgt_col = kwargs.pop('tgt_col')
    self._close = kwargs.pop('close')
    self._high = kwargs.pop('high')
    self._trail_frac = kwargs.pop('trail_frac', None)
    self._sell_eod = kwargs.pop('sell_eod', False)
    self._color = kwargs.pop('color', 'black')
    self._width = kwargs.pop('width', 1)
    self._alpha = kwargs.pop('alpha', 1)
    self._panel = kwargs.pop('panel', 0)
    super().__init__(exmsg='Strategy failure', desc='Strategy')
  
  def _prec(self, df):
    return True
  
  def _preprocess(self, df):
    raise NotImplementedError('_preprocess is not implemented')
  
  def _transform(self, df, verbose):
    if verbose:
      print('Generating Buy/Sell signals for Blender strategy')
    
    ret_df = df.copy()
    ret_df = self._preprocess(ret_df)
    return BuySell(self._tgt_col, self._tgt_col, self._close, self._high, trail_frac=self._trail_frac, sell_eod=self._sell_eod).apply(ret_df)
  
  def _get_or_apply(self, df):
    if self._tgt_col in df.columns:
      return df
    else:
      return self._transform(df, False)
  
  def _indicator(self, df):
    temp = self._get_or_apply(df).copy()
    temp['buy'] = np.where(temp[self._tgt_col] > 0, temp[self._close], np.nan)
    temp['sell'] = np.where(temp[self._tgt_col] < 0, temp[self._close], np.nan)
    
    ret = []
    if len(temp[temp['buy'].notna()]):
      ret.append(mpf.make_addplot(temp['buy'], panel=self._panel, type='scatter', markersize=100, marker='^', width=self._width, alpha=self._alpha))
    if len(temp[temp['sell'].notna()]):
      ret.append(mpf.make_addplot(temp['sell'], panel=self._panel, type='scatter', markersize=100, marker='v', width=self._width, alpha=self._alpha))
    
    return ret

class Blender(Strategy):
  
  def __init__(self, tgt_col, close='Close', high='High', ts='Timestamp', supertrend_multiplier=1.5, donchian_period=21, donchian_thresh=-7, trail_frac=0.01, sell_eod=False, debug=False, **kwargs):
    self._tgt_col = tgt_col
    self._supertrend_multiplier = supertrend_multiplier
    self._donchian_period = donchian_period
    self._donchian_thresh = donchian_thresh
    self._debug = debug
    self._close = close
    self._high = high
    self._ts = ts
    super().__init__(tgt_col=tgt_col, close=close, high=high, trail_frac=trail_frac, sell_eod=sell_eod, **kwargs)
  
  def _preprocess(self, df):
    # Define processing pipeline
    pipeline = pdp.PdPipeline([
      pdp.ColDrop(columns=['VolumeWeighted', 'NumItems', 's'], errors='ignore'),
      ResetIndex(),
      SuperTrend('supertrend', multiplier=self._supertrend_multiplier, as_offset=True),
      #AverageDirectionalIndex('adx', 20),
      DonchianRibbon('donchian', period=self._donchian_period),
      RollingMax('prev_close', self._close, 1),
      EMA('c_ema_52', self._close, 52)
    ])
    
    # Group every two records and aggregate
    agg_raw = IntradayGroups(group_size=2).apply(df)
    
    # Transform 30 and 60 min data via pipeline
    primary = pipeline.apply(df)
    agg = pipeline.apply(agg_raw)
    
    # Join dataframes and filter down to possible setups
    j = primary.join(agg.set_index(self._ts), on=self._ts, rsuffix='_agg', how='left')
    
    j[self._tgt_col] = 0
    
    # Long Signals
    j.loc[
      (j.supertrend > 0) &
      (j.donchian == 10) &
      #(j.adx > 15) &
      (j.supertrend_agg > 0) &
      (j.donchian_agg < 0) &
      (j.donchian_agg >= self._donchian_thresh) &
      (j.Close >= j.c_ema_52_agg),
      self._tgt_col
    ] = 1
    
    return df.join(j.set_index(self._ts)[[self._tgt_col]], on=self._ts, how='left').fillna(method='ffill')