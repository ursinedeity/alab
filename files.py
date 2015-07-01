#!/usr/bin/env python

__author__ = 'N.H.'

import numpy as np
import re
import bisect

def genchrnum(chrom):
  """ Sort by chromosome """
  if chrom:
    num = chrom[3:]
    if   num == 'X': num = 23
    elif num == 'Y': num = 24
    elif num == 'M': num = 25
    else: num = int(num)
  else:
    num = 0
  return num
#====================================================================================
class bedgraph(object):
  """
  Required fields

    chrom      - name of the chromosome or scaffold. Any valid seq_region_name can be used, and chromosome names can be given with or without the 'chr' prefix.
    chromStart - Start position of the feature in standard chromosomal coordinates (i.e. first base is 0).
    chromEnd   - End position of the feature in standard chromosomal coordinates
    dataValue  - Track data values can be integer or real
  """
  bedgraphdtype = np.dtype([('chrom','S5'),('start',int),('end',int),('value',float),('flag','S20')])
  
  def __init__(self,filename=None,usecols=(0,1,2,3,3),**kwargs):
    self.data            = {}
    self.__sorted_keys   = []
    self.itr             = 0
    if not filename is None:
      if isinstance(filename,str):
        from alab.io import loadstream
        f = loadstream(filename)
        readtable = np.genfromtxt(
                            f,
                            dtype=self.bedgraphdtype,
                            usecols=usecols,
                            **kwargs)
        f.close()
      elif isinstance(filename,np.ndarray) or isinstance(filename,list):
        readtable = np.core.records.fromarrays(
                            np.array(filename).transpose()[[usecols]],
                            dtype = self.bedgraphdtype)
      for line in readtable:
        chrom = line['chrom']
        #if np.isnan(line['value']):
          #line['value'] = 0
        if not chrom in self.data:
          self.data[chrom] = []
          self.data[chrom].append(line)
        else:
          #self.data[chrom] = np.append(self.data[chrom],line)
          self.data[chrom].append(line)

      for chrom in self.data:
        self.data[chrom] = np.core.records.fromrecords(self.data[chrom],
                                                      dtype = self.bedgraphdtype)
        self.data[chrom].sort(kind='heapsort',order='start')
      
      self._flush()
      
  #========================================================
  def _flush(self):
    self.__sorted_keys = sorted(self.data.keys(),key=lambda x:genchrnum(x))
    
  def __repr__(self):
    represent = ''
    for chrom in self.__sorted_keys:
      represent += '\n'+chrom+'\twith '+str(len(self.data[chrom]))+' records'
    return represent
  
  def __len__(self):
    length = 0
    for chrom in self.data:
      length += len(self.data[chrom])
    return length
  
  def __getonerec(self,key):
    """For cases a[i], output ith record"""
    if key < 0:
      key += len(self)
    if key > len(self):
      raise IndexError, "The index (%d) is out of range" % key
    for chrom in self.__sorted_keys:
      if key+1 - len(self.data[chrom]) > 0:
        key = key - len(self.data[chrom])
      else:
        return self.data[chrom][key]
      
  #++++++++++++++++++++++++++++++++++++++++++++
  def next(self):
    if self.itr >= len(self):
      raise StopIteration
    self.itr += 1
    return self.__getonerec(self.itr-1)
  
  def __iter__(self):
    self.itr = 0
    return self
  
  def intersect(self,chrom,querystart,querystop):
    """
    Fetch a intersection list with a given interval
    Return a list with all records within the interval
    dtype: numpy record array of (string,int,int,float)
    """
    intersectList = []
    if chrom in self.data:
      i = bisect.bisect(self.data[chrom]['end'],querystart)
      while (i < len(self.data[chrom])) and (self.data[chrom][i]['start'] < querystop):
        intersectList.append(self.data[chrom][i])
        i += 1
    if len(intersectList) == 0:
      return None
    else:
      intersectList = np.core.records.fromrecords(intersectList,dtype=self.bedgraphdtype)
      if intersectList[0]['start'] < querystart:
        intersectList[0]['start'] = querystart
      if intersectList[-1]['end'] > querystop:
        intersectList[-1]['end'] = querystop
      return intersectList
  
  #=========================================================
  def __getitem__(self,key):
    if isinstance(key,int):
      """For case a[1] or a[1:10], return a list of records"""
      return self.__getonerec(key)
    elif isinstance(key,slice):
      if key.step is None: step = 1
      else: step = key.step
      if key.start is None: start = 0
      else: start = key.start
      if key.stop is None: stop = len(self)
      else: stop = key.stop

      if start < 0: start += len(self)
      if stop < 0: stop += len(self)
      if start > len(self) or stop > len(self) :  raise IndexError, "The index out of range"
      records = []
      for i in range(start,stop,step):
        records.append(self.__getonerec(i))
      return np.core.records.fromrecords(records,dtype=self.bedgraphdtype)
      
    elif isinstance(key,tuple):
      """For case a['chr1',3000000:4000000], output average value"""
      chrom = key[0]
      if not chrom in self.data:
        raise KeyError, "Key %s doesn't exist!" % chrom
      try: 
        query = key[1]
      except Exception:
        raise TypeError, "Invalid argument type"
      
      assert isinstance(chrom,str)
      assert isinstance(query,slice)
      if query.start == None: 
        querystart = self.data[chrom][0]['start']
      else: 
        querystart = query.start
      if query.stop == None: 
        querystop = self.data[chrom][-1]['end']
      else: 
        querystop = query.stop
      assert querystop > querystart
      """
      Fetch all intersection with query.start:query.stop 
      """
      queryList = self.intersect(chrom,querystart,querystop)
      value = 0.0
      if not queryList is None:
        #Sum all value
        for rec in queryList:
          value += rec['value'] * (rec['end'] - rec['start'])
  
      return value/(querystop-querystart)
    else:
      raise TypeError, "Invalid argument type"
    
  #=========================================================
  def __setitem__(self,key,value):
    assert isinstance(key,tuple)
    """For case a['chr1',3000000:4000000], input average value"""
    chrom = key[0]
    try: 
      query = key[1]
    except Exception:
      raise TypeError, "Invalid argument type"
    assert isinstance(chrom,str)
    assert isinstance(query,slice)
    
    new = np.array([(chrom,query.start,query.stop,value,'')],
                   dtype=self.bedgraphdtype)
    if not chrom in self.data:
      self.data[chrom] = []
      self.data[chrom].append(new)
      self.data[chrom] = np.core.records.fromrecords(self.data[chrom],dtype = self.bedgraphdtype)
      self._flush()
    else:
      i = bisect.bisect(self.data[chrom]['end'],query.start)
      deletelist = []
      
      if self.data[chrom][i]['start'] < query.start:
        self.data[chrom][i]['end'] = query.start
        i += 1
      
      insertLoc = i
      while (i < len(self.data[chrom])) and (self.data[chrom][i]['end'] < query.stop):
        #print self.data[chrom][i]
        deletelist.append(i)
        i += 1

      if i < len(self.data[chrom]):
        if self.data[chrom][i]['start'] < query.stop:
          self.data[chrom][i]['start'] = query.stop
          
      self.data[chrom] = np.delete(self.data[chrom],deletelist)
      self.data[chrom] = np.insert(self.data[chrom],insertLoc,new)
    #add new data finished
  #=======================================================
  def filter(self,pattern):
    regpattern = re.compile(pattern)
    filterList = []
    for rec in self:
      if regpattern.match(rec['flag']):
        filterList.append(rec)
    return np.core.records.fromrecords(filterList,dtype=self.bedgraphdtype)
      
  #========================================================
  def save(self,filename,bedtype='bedgraph',style='%.8f'):
    """save bed file
    can be bedgraph,bedgraph with flag,bed
    """
    f = open(filename,'w')
    if bedtype == 'bedgraph':
      pattern = '%s\t%d\t%d\t'+style+'\n'
      for chrom in self.__sorted_keys:
        for line in self.data[chrom]:
          f.write(pattern % (chrom,line['start'],line['end'],line['value']))
    elif bedtype == 'bed':
      pattern = '%s\t%d\t%d\t%s\n'
      for chrom in self.__sorted_keys:
        for line in self.data[chrom]:
          f.write(pattern % (chrom,line['start'],line['end'],line['flag']))
    elif bedtype == 'bedgraph+':
      pattern = '%s\t%d\t%d\t'+style+'\t%s\n'
      for chrom in self.__sorted_keys:
        for line in self.data[chrom]:
          f.write(pattern % (chrom,line['start'],line['end'],line['value'],line['flag']))
    elif bedtype == 'bed+':
      pattern = '%s\t%d\t%d\t%s\t'+style+'\n'
      for chrom in self.__sorted_keys:
        for line in self.data[chrom]:
          f.write(pattern % (chrom,line['start'],line['end'],line['flag'],line['value']))
    else:
      raise TypeError, "Invalid argument type %s" % (bedtype)
    f.close


if __name__=='__main__':
  pass
