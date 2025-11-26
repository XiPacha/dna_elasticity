import bubble_parameter as bub
import numpy as np
import matplotlib.pyplot as mpl
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats

def twist_stretch_coupling(start_bp, stop_bp):
    ef=pd.read_table("test_md_h-ris.ser",header=None, sep="\s+")
    df=pd.read_table("test_md_h-twi.ser",header=None, sep="\s+")
    x = df.iloc[:,3:14]
    y = ef.iloc[:,3:14]
   
    x = np.mean(x, axis=1)
    y = np.mean(y, axis=1)
    print np.cov(x,y)
    mpl.plot(x,y,'bo')
    s, i ,r ,p ,std =stats.linregress(x,y)
    mpl.show()
    return s

def func(x, a,P):
    return a*np.exp(-x/P)

def stretch_modulus(start_bp, stop_bp):
    df=pd.read_table("test_md_h-ris.ser",header=None, sep="\s+")
    x = df.iloc[:,3:14]
    r = np.sum(x, axis=1)/10.0
    L = np.mean(r)
    return L*4.114/np.cov(r)


def bending_persistence(start_bp, stop_bp, trajectory, topology):
    end = start_bp +2 
    lengs = []
    bends = []
    while end<=stop_bp:
          L=write_rise(start_bp, end)
          lengs.append(L)
          write_bending(start_bp, end, trajectory, topology)
          B=read_bending(end)
          bends.append(B)
          end+=2
    popt,pcov = curve_fit(func,lengs,bends)
    print popt
    mpl.plot(lengs,bends, 'bo') 
    mpl.show()

def write_rise(start_bp, end):
    df = pd.read_table("test_md_rise.ser", header=None, sep="\s+")
    x  = df.iloc[:,start_bp:end]
    r  = np.sum(x, axis=1)
    L  = np.mean(r)/10.0
    return L

def write_bending(start_bp, end, trajectory, topology):
    ax = bub.axis(trajectory, topology)
    ax.bubble_bending(start_bp, end)



def read_bending(end):
    df = pd.read_table("bending_angle_%s.dat"%(str(end)), sep="\t", header=None)
    x  = df.iloc[:,1]
    B  = x/180.0*np.pi
    B  = np.cos(B)
    B  = np.mean(B)
    return B
