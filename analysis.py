#!/usr/bin/env python

# Copyright (C) 2015 University of Southern California and
#                          Nan Hua
# 
# Authors: Nan Hua
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function
__author__  = "Nan Hua"

__license__ = "GPL"
__version__ = "0.0.1"
__email__   = "nhua@usc.edu"


import numpy as np
import os.path
import h5py
import cPickle as pickle
import alab.matrix
import alab.utils
import alab.files
import copy

class structuresummary(object):
    """
    This class offers a series of methods to study the model structure populations.
    parameters:
    ------------
    target:    the output directory for population structures, containing copy*.hms files
               or can be seen as summary file *.hss
               
    usegrp:    the probablility key used in modeling, e.g. p005j
    nstruct:   number of structures to read
    """
    def __init__(self,target,usegrp=None,nstruct=10000,pid=10,**kwargs):
        if os.path.isdir(target):
            #the target is a valid population structure directory
            if usegrp==None:
                raise RuntimeError, "group key is not specified!"
            firststr    = alab.files.modelstructures(os.path.join(target,"copy0.hms"),[usegrp])
            self.idx    = firststr.idx
            self.genome = firststr.genome
            self.usegrp = usegrp
            self.nbead  = len(self.idx)
            self.nstruct = nstruct
            self.radius  = firststr[-1].r
            print("reading %d structures info in %s,%s"%(nstruct,target,usegrp))
            self._readStructures(target,pid,**kwargs)
        elif os.path.isfile(target) and os.path.splitext(target)[1] == '.hss':
            h5f = h5py.File(target,'r')
            self.idx                   = h5f['idx'][:]
            self.genome                = pickle.loads(h5f['genome'].value)
            self.usegrp                = pickle.loads(h5f['usegrp'].value)
            self.nbead                 = pickle.loads(h5f['nbead'].value)
            self.nstruct               = pickle.loads(h5f['nstruct'].value)
            self.radius                = h5f['radius'][:]
            self.coordinates           = h5f['coordinates'][:]
            self.score                 = h5f['score'][:]
            self.consecutiveViolations = h5f['consecutiveViolations'][:]
            self.contactViolations     = h5f['contactViolations'][:]
            self.intraRestraints       = h5f['intraRestraints'][:]
            self.interRestraints       = h5f['interRestraints'][:]
        else:
            raise RuntimeError, "Invalid filename or file directory!"
        #-
        return None
    #==============================reading
    def _readStructures(self,target,pid,silence=True):
        """
        starting to read structures
        """
        import multiprocessing
        coor_shared       = multiprocessing.Array('d',self.nstruct*self.nbead*2*3) #10000 * 4640 * 3
        consecVio_shared  = multiprocessing.Array('d',self.nstruct)
        contactVio_shared = multiprocessing.Array('d',self.nstruct)
        score_shared      = multiprocessing.Array('d',self.nstruct)
        intrares_shared   = multiprocessing.Array('d',self.nstruct)
        interres_shared   = multiprocessing.Array('d',self.nstruct)
        readpool = []
        for k in range(pid):
            start = k*(self.nstruct/pid)
            end   = (k+1)*(self.nstruct/pid)
            process = multiprocessing.Process(target=self._readInfo,
                                              args=(target,start,end,self.usegrp,
                                                    coor_shared,
                                                    consecVio_shared,
                                                    contactVio_shared,
                                                    score_shared,
                                                    intrares_shared,
                                                    interres_shared,silence))
            process.start()
            readpool.append(process)
        
        for process in readpool:
            process.join()
        self.coordinates = np.frombuffer(coor_shared.get_obj()).reshape((self.nstruct,self.nbead*2,3))
        self.score       = np.frombuffer(score_shared.get_obj())
        self.consecutiveViolations = np.frombuffer(consecVio_shared.get_obj())
        self.contactViolations     = np.frombuffer(contactVio_shared.get_obj())
        self.intraRestraints = np.frombuffer(intrares_shared.get_obj())
        self.interRestraints = np.frombuffer(interres_shared.get_obj())
        return 0
    
    def _readInfo(self,target,start,end,usegrp,
                  coor_shared,
                  consecVio_shared,contactVio_shared,
                  score_shared,
                  intrares_shared,interres_shared,silence=True):
        """
        Read in all information
        """
        arr = np.frombuffer(coor_shared.get_obj())
        modelcoor = arr.reshape((self.nstruct,self.nbead*2,3)) #10000 * 4640 * 3
        for i in range(start,end):
            try:
                if not silence:
                    print(i)
                sts = alab.files.modelstructures(os.path.join(target,'copy'+str(i)+'.hms'),[usegrp])
                
                modelcoor[i][:] = sts[0].xyz
                score_shared[i] = sts[0].score
                
                try:
                    consecVio_shared[i]  = sts[0].consecutiveViolations
                    contactVio_shared[i] = sts[0].contactViolations
                except:
                    consecVio_shared[i]  = np.nan
                    contactVio_shared[i] = np.nan
                try:
                    intrares_shared[i] = sts[0].intraRestraints
                    interres_shared[i] = sts[0].interRestraints
                except:
                    intrares_shared[i] = np.nan
                    interres_shared[i] = np.nan
                
            except RuntimeError:
                print("Can't find result for copy %s , %s at %s" %(str(i),usegrp,os.path.join(target,'copy'+str(i)+'.hms')))
        return 0
    #----------------------------
    
    def getAverageRestraints(self):
        """
        Return average restraints per structure
        """
        return self.interRestraints.mean()+self.intraRestraints.mean()
    
    def getContactMap(self,contactRange=1):
        """
        Return contact matrix format contact heatmap
        """
        from scipy.spatial import distance
        modelmap        = alab.matrix.contactmatrix(self.nbead)
        modelmap.idx    = copy.deepcopy(self.idx)
        modelmap.genome = copy.deepcopy(self.genome)
        modelmap.resolution = None
        n    = self.nbead*2
        dcap = distance.cdist(self.radius,-self.radius)*(contactRange+1)
        cmap = np.zeros((n,n))
        for i in range(self.nstruct):
            if (i+1)*10.0 % self.nstruct == 0:
                print("%.3f %%"%((i+1)*100.0 / self.nstruct))
            #print(i)
            dist   = distance.cdist(self.coordinates[i],self.coordinates[i],'euclidean')
            submap = dist <= dcap
            cmap  += submap
            
        modelmap.matrix = cmap[:n/2,:n/2]+cmap[:n/2,n/2:]+cmap[n/2:,:n/2]+cmap[n/2:,n/2:]
        modelmap.matrix = modelmap.matrix/self.nstruct/2
        
        return modelmap
    
    def getAveragePairwiseDistance(self,form='list'):
        """
        Calculate pairwise distance mean for each pair of beads in the structure population
        
        Parameters
        ----------
        form : the return form of the function
               'list' return the list form
               'matrix' return the matrix form
        """
        pdistMean = np.zeros((2*self.nbead,2*self.nbead))
        for i in range(2*self.nbead):
            for j in range(i+1,2*self.nbead):
                dist = np.linalg.norm(self.coordinates[:,i,:] - self.coordinates[:,j,:],axis=1)
                pdistMean[i,j] = dist.mean()
                pdistMean[j,i] = pdistMean[i,j]
        
        if form == 'list':
            return pdistMean[np.triu_indices(2*self.nbead,1)]
        else:
            return pdistMean
    
    def getBeadRadialPosition(self,beads,nucleusRadius=5000.0):
        """
        Calculate radial position for every bead in the input list beads
        
        Parameters
        ----------
        beads : array-like, list of all beads to calculate
        nucleusRadius : radius of nucleus, default 5000(nm)
        
        Return
        ------
        M*N matrix : radial position for all beads in the input and all structures in population
                     M = len(beads)
                     N = number of structures in population
        """
        allrp = []
        for i in np.array(beads):
            rp = np.linalg.norm(self.coordinates[:,i,:],axis=1)
            allrp.append(rp)
        return np.array(rp)
    
    #==========================saving
    def save(self,filename):
        """
        save all info into disk
        """
        
        if os.path.splitext(filename)[1] != '.hss':
            filename += '.hss'
        h5f = h5py.File(filename,'w')
        h5f.create_dataset('idx',data=self.idx,compression = 'gzip',compression_opts=9)
        h5f.create_dataset('genome',data=pickle.dumps(self.genome))
        h5f.create_dataset('usegrp',data=pickle.dumps(self.usegrp))
        h5f.create_dataset('nbead',data=pickle.dumps(self.nbead))
        h5f.create_dataset('nstruct',data=pickle.dumps(self.nstruct))
        h5f.create_dataset('radius',data=self.radius,compression = 'gzip',compression_opts=9)
        h5f.create_dataset('coordinates',data=self.coordinates,compression = 'gzip',compression_opts=9)
        h5f.create_dataset('score',data=self.score,compression = 'gzip',compression_opts=9)
        h5f.create_dataset('consecutiveViolations',data=self.consecutiveViolations,compression = 'gzip',compression_opts=9)
        h5f.create_dataset('contactViolations',data=self.contactViolations,compression = 'gzip',compression_opts=9)
        h5f.create_dataset('intraRestraints',data=self.intraRestraints,compression = 'gzip',compression_opts=9)
        h5f.create_dataset('interRestraints',data=self.interRestraints,compression = 'gzip',compression_opts=9)
        h5f.close()
        
        return 0
    
if __name__=='__main__':
    pass

