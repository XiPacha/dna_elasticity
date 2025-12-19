import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import pandas as pd

def linear(x,m):
    return 2*x*m


def bend_modulus():
    df1=pd.read_csv("bending_angle_5.dat", header=None,sep="\s+")
    df2=pd.read_csv("bending_angle_7.dat", header=None,sep="\s+")
    df3=pd.read_csv("bending_angle_9.dat", header=None,sep="\s+")
    df4=pd.read_csv("bending_angle_11.dat",header=None,sep="\s+")
    df1,df2,df3,df4= np.asarray(df1.iloc[:,1]), np.asarray(df2.iloc[:,1]), np.asarray(df3.iloc[:,1]), np.asarray(df4.iloc[:,1])
    df1,df2,df3,df4 = df1/180.0*np.pi, df2/180.0*np.pi, df3/180.0*np.pi, df4/180.0*np.pi
    df1,df2,df3,df4 = np.mean(df1**2),np.mean(df2**2),np.mean(df3**2),np.mean(df4**2)
    L1,L2,L3,L4     = 2*0.34, 4*0.34, 6*0.34, 8*0.34
    angle = np.asarray([df1,df2,df3,df4])
    L = np.asarray([L1,L2,L3,L4])
    popt, pcov = curve_fit(linear, L, angle)
    A=1/popt

    print(A)

