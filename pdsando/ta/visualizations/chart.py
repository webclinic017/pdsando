import mplfinance as mpf
import pdpipe as pdp

from pdsando.ta.pipeline.transforms import SetIndex
from pdsando.ta.pipeline.indicators import Indicator
from pdsando.ta.pipeline.strategies import Strategy

class Chart:
  def __init__(self, data, open='Open', high='High', low='Low', close='Close', volume='Volume', ts='Timestamp'):
    self._data = data
    self._indicators = []
    
    self._open = open
    self._high = high
    self._low = low
    self._close = close
    self._volume = volume
    self._ts = ts
  
  def add_indicator(self, *args):
    for ind in args:
      self._indicators.append(ind)
  
  def draw(self, session_breaks=True, figsize=(35,15), savefig=None):
    vlines = []
    if session_breaks:
      min_times = self._data.groupby([self._data['Timestamp'].dt.year, self._data['Timestamp'].dt.month, self._data['Timestamp'].dt.day])['Timestamp'].transform('min')
      vlines = list(self._data[(self._data['Timestamp'] == min_times)]['Timestamp'])
    
    # Generate indicator data
    apd = []
    for x in self._indicators:
      try:
        apd.extend(x._indicator(self._data))
      except AttributeError as ae:
        print(ae)
      except NotImplementedError as nie:
        print(nie)
    
    # Prepare data for charting - data needs to meet expected schema for mplfinance.
    df = SetIndex(self._ts).apply(self._data)
    
    # Draw plot
    if savefig:
      mpf.plot(df, type='candlestick', style='yahoo', figsize=figsize, addplot=apd, savefig=savefig, tight_layout=True, vlines=dict(vlines=vlines, alpha=0.35, linewidths=1))
    else:
      mpf.plot(df, type='candlestick', style='yahoo', figsize=figsize, addplot=apd, vlines=dict(vlines=vlines, alpha=0.35, linewidths=1))
  
  def draw_pipeline(self, pipeline, session_breaks=True, figsize=(35,15), savefig=None):
    df = pipeline.apply(self._data)
    vlines = []
    if session_breaks:
      min_times = df.groupby([df['Timestamp'].dt.year, df['Timestamp'].dt.month, df['Timestamp'].dt.day])['Timestamp'].transform('min')
      vlines = list(df[(df['Timestamp'] == min_times)]['Timestamp'])
    
    # Generate indicator data
    apd = []
    for x in [ x for x in pipeline if isinstance(x, Indicator) or isinstance(x, Strategy) ]:
      try:
        apd.extend(x._indicator(df))
      except AttributeError as ae:
        print(ae)
      except NotImplementedError as nie:
        print(nie)
    
    # Prepare data for charting - data needs to meet expected schema for mplfinance.
    df = SetIndex(self._ts).apply(df)
    
    # Draw plot
    if savefig:
      mpf.plot(df, type='candlestick', style='yahoo', figsize=figsize, addplot=apd, savefig=savefig, tight_layout=True, vlines=dict(vlines=vlines, alpha=0.35, linewidths=1))
    else:
      mpf.plot(df, type='candlestick', style='yahoo', figsize=figsize, addplot=apd, vlines=dict(vlines=vlines, alpha=0.35, linewidths=1))