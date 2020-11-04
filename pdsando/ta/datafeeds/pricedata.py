from pandas import DataFrame

class PriceData(DataFrame):
  def __init__(self, *args, **kwargs):
    # Validate inputs
    valid_timespans = ('second', 'minute', 'hour', 'day', 'month', 'year')
    if kwargs['timespan'] not in valid_timespans:
      raise ValueError(f'Timespan must be one of: {valid_timespans}')
    if int(kwargs['multiplier']) <= 0:
      raise ValueError(f'Multiplier must be a valid integer greater than 0')
    
    # Init Pandas DataFrame
    super().__init__(*args, **{ k:kwargs[k] for k in kwargs if k not in ['timespan', 'multiplier', 'source'] })
    
    # Store additional details specific to PriceData
    self._timespan = kwargs['timespan']
    self._multiplier = int(kwargs['multiplier'])
    self._source = kwargs['source']
  
  @property
  def timespan(self): return self._timespan
  @property
  def multiplier(self): return self._multiplier
  @property
  def source(self): return self._source
  
  def copy(self, **kwargs):
    return PriceData(super().copy(**kwargs), timespan=self._timespan, multiplier=self._multiplier, source=self._source)
  
  # Up/downscale data resolution
  def match_resolution(self, model_data):
    if model_data.timespan != self._timespan:
      raise NotImplementedError('Currently cannot support matching different timespan (source: {} | target: {})'.format(self._timespan, model_data.timespan))
    if model_data.multiplier == self._multiplier:
      return
    
    if model_data.multiplier > self._multiplier:
      temp = self.join(model_data.set_index('Timestamp'), on='Timestamp', rsuffix='_model', how='inner')
    else:
      temp = self.join(model_data.set_index('Timestamp'), on='Timestamp', rsuffix='_model', how='right').sort_values(by='Timestamp').fillna(method='ffill')
    
    return temp[list(self.columns)].copy().sort_values(by='Timestamp').reset_index(drop=True)