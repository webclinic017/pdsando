import pdpipe as pdp
from pdpipe import PdPipelineStage
import pandas_market_calendars as mcal

class RemoveNonMarketHours(PdPipelineStage):
  _EXC_MSG = 'RemoveNonMarketHours failure'
  _DESC = 'RemoveNonMarketHours'
  
  def __init__(self, ts='Timestamp', **kwargs):
    self._ts = ts
    super_kwargs = {
      'exmsg': RemoveNonMarketHours._EXC_MSG,
      'desc' : RemoveNonMarketHours._DESC
    }
    super_kwargs.update(**kwargs)
    super().__init__(**super_kwargs)
  
  def _prec(self, df):
    return True
  
  def _transform(self, df, verbose):
    if verbose:
      print('Removing non-market hours from dataset.')
    
    return df[
      (df[self._ts].dt.hour * 60 + df[self._ts].dt.minute >= 570) # 9:30 AM EST
      #(df[self._ts].dt.hour * 60 + df[self._ts].dt.minute >= 540) # 9:00 AM EST
      &
      (df[self._ts].dt.hour * 60 + df[self._ts].dt.minute < 960) # 4:00 PM EST
      &
      (df[self._ts].dt.weekday >= 0)
      &
      (df[self._ts].dt.weekday <= 4)
    ]