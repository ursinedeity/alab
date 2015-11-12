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

# Prerequests:
# IMP 2.4 is required for this module

__author__ = 'N.H.'

import alab.utils
import time
import numpy as np
import IMP
import IMP.core
import IMP.container
import IMP.algebra
import IMP.atom
import random

def beadDistanceRestraint(model,chain,bead1,bead2,dist,kspring=1):
    """
        get distance upper bound restraint to bead1 and bead2
        Return restraint 
        Parameters:
        -----------
        model:      IMP.Model class
        chain:      IMP.container.ListSingletonContainer class
        bead1,bead2:bead id
        dist:       distance upper bound
        kspring:    harmonic constant k
    """
    restraintName = "Bead (%d,%d):%f k = %f" % (bead1,bead2,dist,kspring)
    ds = IMP.core.SphereDistancePairScore(IMP.core.HarmonicUpperBound(dist,kspring))
    pr = IMP.core.PairRestraint(model,ds,IMP.ParticlePair(chain.get_indexes()[bead1],chain.get_indexes()[bead2]),restraintName)
    return pr

#-------------------consecutive bead restraints --------------------------
def consecutiveDistanceByProbability(r1,r2,p,xcontact=2):
    """
        Upper bound distance constraints for consecutive domains
        return surface to surface distance.
        parameters:
        -----------
        r1,r2:     Radius for beads
        p:         Probability for contact
        xcontact:  scaling of (r1+r2) where a contact is defined. By default, 
                   center to center distance D = 2*(r1+r2) is defined as contact.
    """
    if p > 0:
        d = (r1+r2)*(1. + (xcontact**3-1)/p)**(1./3.)
    else:
        d = 100*(r1+r2) # just a big number
    return d-r1-r2 # surface to surface distance

def consecutiveBeadRestraints(model,chain,probmat,beadrad,contactRange=1,lowprob=0.1):
    """
        calculate distance constraints to consecutive beads
        Parameters:
        -----------
        model:      IMP.Model class
        chain:      IMP.container.ListSingletonContainer class
        probmat:    alab.matrix.contactmatrix class for probablility matrix
        beadrad:    list like, radius of each bead
        contactRange scale of (r1+r2) where a contact is defined
        lowprob:    Min probility for consecutive beads
    """
    consecRestraints = []
    nbead = len(probmat)
    for i in range(nbead-1):
        if probmat.idx[i]['chrom'] != probmat.idx[i+1]['chrom']:
            continue
        p = max(probmat.matrix[i,i+1],lowprob)
        b1 = i;b2 = i+1
        b3 = b1 + nbead
        b4 = b2 + nbead
        #calculate upper bound for consecutive domains
        consecDist = consecutiveDistanceByProbability(beadrad[b1],beadrad[b2],p,contactRange+1)
           
        # set k = 10 to be strong interaction
        rs1 = beadDistanceRestraint(model,chain,b1,b2,consecDist,kspring=10) 
        rs2 = beadDistanceRestraint(model,chain,b3,b4,consecDist,kspring=10) 
           
        #push restraint into list
        consecRestraints.append(rs1)
        consecRestraints.append(rs2)
        
        if i>0 and probmat.idx[i]['chrom'] == probmat.idx[i-1]['chrom'] and probmat.idx[i]['flag']!="domain" and probmat.idx[i-1]!="gap":
            p = max(probmat.matrix[i-1,i+1],lowprob)
            b1 = i-1;b2 = i+1
            b3 = b1 + nbead
            b4 = b2 + nbead
            consecDist = consecutiveDistanceByProbability(beadrad[b1],beadrad[b2],p,contactRange+1)
            rs1 = beadDistanceRestraint(model,chain,b1,b2,consecDist,kspring=10) 
            rs2 = beadDistanceRestraint(model,chain,b3,b4,consecDist,kspring=10) 
            consecRestraints.append(rs1)
            consecRestraints.append(rs2)
        #---
    #-------
    
    return consecRestraints
#=============================end consecutive bead restraints
#-----------------------------chromosome territory functions
def CondenseChromosome(chain, probmat, genome, rchrs, rrange=0.5):
    """
        Collapse chains around centromere beads
        parameters:
        -----------
        chain:     IMP.container.ListSingletonContainer class
        probmat:   alab.matrix.contactmatrix class for probablility matrix
        genome:    alab.utils.genome class, containing genome information
        rchrs:     chromosome territory radius
        rrange:    scale parameter in [0,1] for the radius limit
    """
    import random
    nbead = len(probmat)
    i = -1
    for chrom in genome.info['chrom']:
        i += 1
        #find centromere
        cenbead = np.flatnonzero((probmat.idx['chrom'] == chrom) & (probmat.idx['flag'] == 'CEN'))[0]
        p0A=IMP.core.XYZ(chain.get_particles()[cenbead]) #fetch indexes
        p0B=IMP.core.XYZ(chain.get_particles()[cenbead+nbead])
        coorA = p0A.get_coordinates()
        coorB = p0B.get_coordinates()
        rlimit = rchrs[i]*rrange
        for j in np.flatnonzero(probmat.idx['chrom'] == chrom):
            p1A=IMP.core.XYZ(chain.get_particles()[j])
            p1B=IMP.core.XYZ(chain.get_particles()[j+nbead])
            dx=(2*random.random()-1)*rlimit
            dy=(2*random.random()-1)*rlimit
            dz=(2*random.random()-1)*rlimit
            randA = coorA
            randA[0] += dx
            randA[1] += dy
            randA[2] += dz
            dx=(2*random.random()-1)*rlimit
            dy=(2*random.random()-1)*rlimit
            dz=(2*random.random()-1)*rlimit
            randB = coorB
            randB[0] += dx
            randB[1] += dy
            randB[2] += dz
            p1A.set_coordinates(randA) #placed nearby cen
            p1B.set_coordinates(randB) #placed nearby cen
        #--
    #--
#=============================end chromosome terriory
#-----------------------------probmat restraints
def minPairRestraints(model,chain,bpair,dist,minnum,kspring = 1):
    """
        Return restraint decorater of min pair restraints
        for minnum out of bpairs are satisfied 
        Parameters:
        -----------
        model:       IMP.Model class
        chain:       IMP.container.ListSingletonContainer class
        bpair:       tuple list of contact pair candidates
        upperdist:   distance upperbound for contact
        minnum:      minimun number of pairs required to satisify
        contactRange:scale of (r1+r2) where a contact is defined   
    """
    ambi = IMP.container.ListPairContainer(model)
    for p in bpair:
        p0 = chain.get_particles()[p[0]]
        p1 = chain.get_particles()[p[1]]
        pair = IMP.ParticlePair(p0,p1)
        ambi.add_particle_pair(pair)
    ds = IMP.core.SphereDistancePairScore(IMP.core.HarmonicUpperBound(dist,kspring))
    minpr = IMP.container.MinimumPairRestraint(ds,ambi,minnum)
    return minpr

def fmaxRestraints(model,chain,probmat,beadrad,contactRange):
    """
        return restraints list for prob=1.0
        parameters:
        -----------
        model:       IMP.Model class
        chain:       IMP.container.ListSingletonContainer class
        probmat:     alab.matrix.contactmatrix class for probablility matrix
        beadrad:     list like, radius of each bead
        contactRange:scale of (r1+r2) where a contact is defined
    """
    fmaxrs = []
    nbead = len(probmat)
    for i in range(nbead):
        for j in range(i+1,nbead):
            if probmat.matrix[i,j] <= 0.999:
                continue
            if probmat.idx[i]['chrom'] == probmat.idx[j]['chrom']: #intra
                if j-i > 1:
                    rs1 = beadDistanceRestraint(model,chain,i,j,contactRange*(beadrad[i]+beadrad[j]))
                    rs2 = beadDistanceRestraint(model,chain,i+nbead,j+nbead,contactRange*(beadrad[i]+beadrad[j]))
                    fmaxrs.append(rs1)
                    fmaxrs.append(rs2)
            else: #inter
                bpair = [(i,j),(i,j+nbead),(i+nbead,j),(i+nbead,j+nbead)] #bead pair
                minprrs = minPairRestraints(model,chain,bpair,contactRange*(beadrad[i]+beadrad[j]),minnum=2)
                fmaxrs.append(minprrs)
            #--
        #--
    #--
    return fmaxrs
#=============================end probmat restraints

#-----------------------------modeling steps
def cgstep(model,sf,step,silent=False):
    """
        perform conjugate gradient on model using scoring function sf
    """
    t0 = time.time()
    o = IMP.core.ConjugateGradients(model)
    o.set_scoring_function(sf)
    o.set_log_level(IMP.SILENT)
    s = o.optimize(step)
    if not silent:
        print 'CG',step,'steps done @',alab.utils.timespend(t0)
    return s

def mdstep(model,chain,sf,t,step,silent=False):
    t0 = time.time()
    xyzr = chain.get_particles()
    o    = IMP.atom.MolecularDynamics(model)
    o.set_scoring_function(sf)
    o.set_log_level(IMP.SILENT)
    md   = IMP.atom.VelocityScalingOptimizerState(model,xyzr,t)
    o.add_optimizer_state(md)
    s    = o.optimize(step)
    o.remove_optimizer_state(md)
    if not silent:
        print 'MD',step,'steps done @',alab.utils.timespend(t0)
    return s

def mdstep_withChromosomeTerriory(model,chain,restraints,probmat,genome,rchrs,t,step):
    """
        perform an mdstep with chromosome terriory restraint
        parameters:
        -----------
        chain:     IMP.container.ListSingletonContainer class
        probmat:   alab.matrix.contactmatrix class for probablility matrix
        genome:    alab.utils.genome class, containing genome information
        rchrs:     chromosome territory radius
        t:         temperature
        step:      optimization steps
    """
    t0 = time.time()
    nbead = len(probmat)
    chrContainers=[]
    for chrom in genome.info['chrom']:
        chromContainer1=IMP.container.ListSingletonContainer(model,'Container %s s1'%chrom)
        chromContainer2=IMP.container.ListSingletonContainer(model,'Container %s s2'%chrom)
        for j in np.flatnonzero(probmat.idx['chrom'] == chrom):
            p=chain.get_particle(j)
            chromContainer1.add_particle(p)
            p=chain.get_particle(j+nbead)
            chromContainer2.add_particle(p)
        chrContainers.append(chromContainer1)
        chrContainers.append(chromContainer2)
    # set each chromosome to different containers
    for st in range(step/10):
        ctRestraintSet = IMP.RestraintSet(model)
        for i in range(len(genome.info['chrom'])):
            comxyz = centerOfMass(chrContainers[2*i])
            comc   = IMP.algebra.Vector3D(comxyz[0],comxyz[1],comxyz[2])
            ub     = IMP.core.HarmonicUpperBound(rchrs[i],0.2)
            ss     = IMP.core.DistanceToSingletonScore(ub,comc)
            ct_rs  = IMP.container.SingletonsRestraint(ss,chrContainers[2*i])
            
            ctRestraintSet.add_restraint(ct_rs)
            #another one
            comxyz = centerOfMass(chrContainers[2*i+1])
            comc   = IMP.algebra.Vector3D(comxyz[0],comxyz[1],comxyz[2])
            ub     = IMP.core.HarmonicUpperBound(rchrs[i],0.2)
            ss     = IMP.core.DistanceToSingletonScore(ub,comc)
            ct_rs  = IMP.container.SingletonsRestraint(ss,chrContainers[2*i+1])
            
            ctRestraintSet.add_restraint(ct_rs)
        sf = IMP.core.RestraintsScoringFunction([restraints,ctRestraintSet])
        s = mdstep(model,chain,sf,t,10,silent=True)
    #---
    #s = mdstep(model,chain,sf,t,1000)
    print 'CT-MD',step,'steps done @',alab.utils.timespend(t0)
    return s

def SimulatedAnnealing(model,chain,sf,hot,cold,nc=10,nstep=500):
    """
        perform a cycle of simulated annealing from hot to cold
    """
    t0 = time.time()
    dt = (hot-cold)/nc
    for i in range(nc):
        t = hot-dt*i
        mdstep(model,chain,sf,t,nstep)
        print "      Temp=%d Step=%d Time=%.3f"%(t,nstep,alab.utils.timespend(t0))
    mdstep(model,chain,sf,cold,nstep)
    print "      Temp=%d Step=%d Time=%.3f"%(cold,nstep,alab.utils.timespend(t0))
    cgstep(model,sf,100)
#=============================end modeling steps
def centerOfMass(chain):
    xyzm = np.zeros((len(chain.get_particles()),4))
    i = -1
    for p in chain.get_particles():
        i += 1
        pattr = IMP.core.XYZR(p)
        xyzm[i,3] = pattr.get_radius()**3
        xyzm[i,0] = pattr.get_x()*xyzm[i,3]
        xyzm[i,1] = pattr.get_y()*xyzm[i,3]
        xyzm[i,2] = pattr.get_z()*xyzm[i,3]
    #---
    mass = sum(xyzm[:,3])
    return (sum(xyzm[:,0])/mass,sum(xyzm[:,1])/mass,sum(xyzm[:,2])/mass)
def savepym(filename,chain):
    pymfile = IMP.display.PymolWriter(filename)
    g = IMP.core.XYZRsGeometry(chain)
    g.set_name('beads')
    g.set_color(IMP.display.Color(1,1,1))
    pymfile.add_geometry(g)

def savepym_withChromosome(filename,model,chain,probmat,genome):
    pymfile = IMP.display.PymolWriter(filename)
    nbead = len(probmat)
    for chrom in genome.info['chrom']:
        chromContainer1=IMP.container.ListSingletonContainer(model,'Container %s s1'%chrom)
        chromContainer2=IMP.container.ListSingletonContainer(model,'Container %s s2'%chrom)
        for j in np.flatnonzero(probmat.idx['chrom'] == chrom):
            p=chain.get_particle(j)
            chromContainer1.add_particle(p)
            p=chain.get_particle(j+nbead)
            chromContainer2.add_particle(p)
        color = IMP.display.Color(random.random(),random.random(),random.random())
        g1 = IMP.core.XYZRsGeometry(chromContainer1)
        g1.set_name(chrom+' s1')
        g1.set_color(color)
        pymfile.add_geometry(g1)
        g2 = IMP.core.XYZRsGeometry(chromContainer2)
        g2.set_name(chrom+' s2')
        g2.set_color(color)
        pymfile.add_geometry(g2)
