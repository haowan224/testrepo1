#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 08:33:22 2023

@author: haowan
"""

import matplotlib.pyplot as plt
import numpy as np
from pylab import polyfit 
from matplotlib.ticker import MultipleLocator
import matplotlib.transforms
import matplotlib.path
from matplotlib.collections import LineCollection
import matplotlib.pylab as pl
from scipy.interpolate import interp1d


def rainbowarrow(ax, start, end, cmap="viridis", n=50,lw=3):
    cmap = plt.get_cmap(cmap,n)
    # Arrow shaft: LineCollection
    x = np.linspace(start[0],end[0],n)
    y = np.linspace(start[1],end[1],n)
    points = np.array([x,y]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1],points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, linewidth=lw)
    lc.set_array(np.linspace(0,1,n))
    ax.add_collection(lc)
    # Arrow head: Triangle
    tricoords = [(0,-0.4),(0.5,0),(0,0.4),(0,-0.4)]
    angle = np.arctan2(end[1]-start[1],end[0]-start[0])
    rot = matplotlib.transforms.Affine2D().rotate(angle)
    tricoords2 = rot.transform(tricoords)
    tri = matplotlib.path.Path(tricoords2, closed=True)
    ax.scatter(end[0],end[1], c=1, s=(2*lw)**2, marker=tri, cmap=cmap,vmin=0)
    ax.autoscale_view()


def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b


### configuration entropy
#Kb = 1.380649e-23  ####J/K

Kb = 8.6173324e-5  ####eV/K
C = 1/9
S_con_1 = -Kb*(C*np.log(C) + (1-C)*np.log(1-C)) 
T = 298.15
#print(-S_con_1*T) 

C = 2/9
S_con_2 = -Kb*(C*np.log(C) + (1-C)*np.log(1-C)) 

#print(-S_con_2*T) 


C = 3/9
S_con_3 = -Kb*(C*np.log(C) + (1-C)*np.log(1-C)) 

#print(-S_con_3*T) 


C = 4/9
S_con_4 = -Kb*(C*np.log(C) + (1-C)*np.log(1-C)) 

#print(-S_con_4*T) 

C = 5/25
S_con_5 = -Kb*(C*np.log(C) + (1-C)*np.log(1-C)) 

#print(-S_con_5*T) 
C = 8/25
S_con_8 = -Kb*(C*np.log(C) + (1-C)*np.log(1-C)) 



Gamma_Cu100 = np.array([0.208177303])
Gamma_Cu100_CO = np.array([0.199528549+0.006564519-S_con_1*T/(7.806*7.806)])
Gamma_Cu100_2CO = np.array([0.191355723+2*0.006564519-S_con_2*T/(7.806*7.806)])
Gamma_Cu100_3CO = np.array([0.183740882 + 3*0.006564519-S_con_3*T/(7.806*7.806)])
Gamma_Cu100_4CO = np.array([0.17827592 + 4*0.006564519-S_con_4*T/(7.806*7.806)])

Gamma_NiO100 = np.array([0.276166872])
Gamma_NiO100_CO = np.array([0.245812816+0.005503909-S_con_1*T/(8.373*8.373)])
Gamma_NiO100_2CO = np.array([0.215266123+2*0.005503909-S_con_2*T/(8.373*8.373)])
Gamma_NiO100_3CO = np.array([0.2021393+3*0.005503909-S_con_3*T/(8.373*8.373)])


Gamma_Ni100 = np.array([0.228623002])
Gamma_Ni100_CO = np.array([0.204042139+0.00698817-S_con_1*T/(57.198969)])
Gamma_Ni100_2CO = np.array([0.180597661+2*0.00698817-S_con_2*T/(57.198969)])
Gamma_Ni100_3CO = np.array([0.156593732+3*0.00698817-S_con_3*T/(57.198969)])




Gamma_1MLCuNiO100 = np.array([0.301991899])
Gamma_2MLCuNiO100 = np.array([0.327877469])
Gamma_3MLCuNiO100 = np.array([0.35793225])

Gamma_4MLCuNiO100 = np.array([0.375039086])
#Gamma_4MLCuNiO100_CO = np.array([0.366865782])
#Gamma_4MLCuNiO100_2CO = np.array([0.359504304])
#Gamma_4MLCuNiO100_3CO = np.array([0.352280424])
#Gamma_4MLCuNiO100_4CO = np.array([0.355816685])
Gamma_4MLCuNiO100_CO = np.array([0.372369691-S_con_1*T/(8.373*8.373)])
Gamma_4MLCuNiO100_2CO = np.array([0.370512121-S_con_2*T/(8.373*8.373)])
Gamma_4MLCuNiO100_3CO = np.array([0.36879215-S_con_3*T/(8.373*8.373)])


Gamma_3MLCuNiO100_CO = np.array([0.350213018+0.005503909-S_con_1*T/(8.373*8.373)])
Gamma_3MLCuNiO100_2CO = np.array([0.342191072+2*0.005503909-S_con_2*T/(8.373*8.373)])
Gamma_3MLCuNiO100_3CO = np.array([0.334031527+3*0.005503909-S_con_3*T/(8.373*8.373)])


Gamma_2MLCuNiO100_CO = np.array([0.317378764+0.005503909-S_con_1*T/(8.373*8.373)])
Gamma_2MLCuNiO100_2CO = np.array([0.308256035+2*0.005503909-S_con_2*T/(8.373*8.373)])
Gamma_2MLCuNiO100_3CO = np.array([0.301431188+3*0.005503909-S_con_3*T/(8.373*8.373)])

##### 3 5 big unitcell 
Gamma_2MLCuNiO100_35 = np.array([0.3122483])
Gamma_2MLCuNiO100_5CO_35 = np.array([0.296778678+5*0.005503909*(8.373*8.373)/(12.559*12.559)-S_con_5*T/(12.559*12.559)])
Gamma_2MLCuNiO100_8CO_35 = np.array([0.286691723+8*0.005503909*(8.373*8.373)/(12.559*12.559)-S_con_8*T/(12.559*12.559)])


#print(Gamma_2MLCuNiO100_5CO_35)
Gamma_1MLCuNiO100_CO = np.array([0.294644181-S_con_1*T/(8.373*8.373)])
Gamma_1MLCuNiO100_2CO = np.array([0.289993378-S_con_2*T/(8.373*8.373)])
Gamma_1MLCuNiO100_3CO = np.array([0.286553435-S_con_3*T/(8.373*8.373)])

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
lw=5
al=0.3

datacolor  = ['k','r','indigo','mediumspringgreen','navy','b','plum','m','gray','coral', 'tan']


n = 11
colors2 = pl.cm.Blues(np.linspace(0, 1, n))
########\Delat E ######  CoN3CH


data_marker_size = 12
text_size = 22

#ax.annotate('Cu(100)', xy=(0,0), xytext=(-85,5), size= text_size, textcoords='offset points', color=datacolor[0])


### Cu 100
plt.plot(0,(Gamma_Cu100[0]*1000 ),marker='o',color=colors2[10], ls='', markersize=data_marker_size)
plt.plot(1/9,(Gamma_Cu100_CO[0]*1000 ),marker='o',color=colors2[10], ls='', markersize=data_marker_size)
plt.plot(2/9,(Gamma_Cu100_2CO[0]*1000 ),marker='o',color=colors2[10], ls='', markersize=data_marker_size)
plt.plot(3/9,(Gamma_Cu100_3CO[0]*1000),marker='o',color=colors2[10], ls='', markersize=data_marker_size)
#plt.plot(4,(Gamma_Cu100_4CO[0]*1000 - Gamma_Cu100[0]*1000),marker='o',color=datacolor[4], ls='', markersize=data_marker_size)
print((Gamma_Cu100_CO[0]*1000))
print((Gamma_Cu100_2CO[0]*1000))
print((Gamma_Cu100_3CO[0]*1000))

### NiO100
#ax.annotate('NiO(100)', xy=(0,50), xytext=(-85,5), size= text_size, textcoords='offset points', color=datacolor[1])

plt.plot(0/9,Gamma_NiO100[0]*1000,marker='*',color=datacolor[1], ls='', markersize=data_marker_size)
plt.plot(1/9, Gamma_NiO100_CO[0]*1000,marker='*',color=datacolor[1], ls='', markersize=data_marker_size)
plt.plot(2/9, Gamma_NiO100_2CO[0]*1000,marker='*',color=datacolor[1], ls='', markersize=data_marker_size)
plt.plot(3/9, Gamma_NiO100_3CO[0]*1000,marker='*',color=datacolor[1], ls='', markersize=data_marker_size)


print((Gamma_NiO100_CO[0]*1000 ))
print((Gamma_NiO100_2CO[0]*1000))
print((Gamma_NiO100_3CO[0]*1000))

### Ni100
#ax.annotate('Ni(100)', xy=(0,50), xytext=(-85,5), size= text_size, textcoords='offset points', color=datacolor[1])

plt.plot(0/9,Gamma_Ni100[0]*1000,marker='s',color=datacolor[1], ls='', markersize=data_marker_size)
plt.plot(1/9, Gamma_Ni100_CO[0]*1000,marker='s',color=datacolor[1], ls='', markersize=data_marker_size)
plt.plot(2/9, Gamma_Ni100_2CO[0]*1000,marker='s',color=datacolor[1], ls='', markersize=data_marker_size)
plt.plot(3/9, Gamma_Ni100_3CO[0]*1000,marker='s',color=datacolor[1], ls='', markersize=data_marker_size)

print((Gamma_Ni100_CO[0]*1000 ))
print((Gamma_Ni100_2CO[0]*1000 ))
print((Gamma_Ni100_3CO[0]*1000 ))

#### 1MLCu@NiO
#ax.annotate('1MLCu \n @NiO', xy=(0,76), xytext=(-85,5), size= text_size, textcoords='offset points', color=datacolor[2])

plt.plot(0,Gamma_1MLCuNiO100[0]*1000,marker='^',color=colors2[2], ls='', markersize=data_marker_size)
plt.plot(1/9,Gamma_1MLCuNiO100_CO[0]*1000,marker='^',color=colors2[2], ls='', markersize=data_marker_size)
plt.plot(2/9,Gamma_1MLCuNiO100_2CO[0]*1000,marker='^',color=colors2[2], ls='', markersize=data_marker_size)
plt.plot(3/9,Gamma_1MLCuNiO100_3CO[0]*1000,marker='^',color=colors2[2], ls='', markersize=data_marker_size)
print((Gamma_1MLCuNiO100_CO[0]*1000))
print((Gamma_1MLCuNiO100_2CO[0]*1000))
print((Gamma_1MLCuNiO100_3CO[0]*1000))
#### 2MLCu@NiO
#ax.annotate('2MLCu \n @NiO', xy=(0,107), xytext=(-85,5), size= text_size, textcoords='offset points', color=datacolor[3])

plt.plot(0/9,Gamma_2MLCuNiO100[0]*1000,marker='p',color=colors2[4], ls='', markersize=data_marker_size)
plt.plot(1/9,Gamma_2MLCuNiO100_CO[0]*1000,marker='p',color=colors2[4], ls='', markersize=data_marker_size)
plt.plot(2/9,Gamma_2MLCuNiO100_2CO[0]*1000,marker='p',color=colors2[4], ls='', markersize=data_marker_size)
plt.plot(3/9,Gamma_2MLCuNiO100_3CO[0]*1000,marker='p',color=colors2[4], ls='', markersize=data_marker_size)

#### 3MLCu@NiO
#ax.annotate('3MLCu \n @NiO', xy=(0,136), xytext=(-85,5), size= text_size, textcoords='offset points', color=datacolor[4])

plt.plot(0/9,Gamma_3MLCuNiO100[0]*1000,marker='x',color=colors2[6], ls='', markersize=data_marker_size)
plt.plot(1/9,Gamma_3MLCuNiO100_CO[0]*1000,marker='x',color=colors2[6], ls='', markersize=data_marker_size)
plt.plot(2/9,Gamma_3MLCuNiO100_2CO[0]*1000,marker='x',color=colors2[6], ls='', markersize=data_marker_size)
plt.plot(3/9,Gamma_3MLCuNiO100_3CO[0]*1000,marker='x',color=colors2[6], ls='', markersize=data_marker_size)

#### 4MLCu@NiO
#ax.annotate('4MLCu \n @NiO', xy=(0,165), xytext=(-85,5), size= text_size, textcoords='offset points', color=datacolor[5])

plt.plot(0/9,Gamma_4MLCuNiO100[0]*1000,marker='>',color=colors2[8], ls='', markersize=data_marker_size)
plt.plot(1/9,Gamma_4MLCuNiO100_CO[0]*1000,marker='>',color=colors2[8], ls='', markersize=data_marker_size)
plt.plot(2/9,Gamma_4MLCuNiO100_2CO[0]*1000,marker='>',color=colors2[8], ls='', markersize=data_marker_size)
plt.plot(3/9,Gamma_4MLCuNiO100_3CO[0]*1000,marker='>',color=colors2[8], ls='', markersize=data_marker_size)


X = [0, 1/9, 2/9, 3/9] 
Cu100 = [Gamma_Cu100[0]*1000 , 205.94598346813729, 204.2614111728435, 203.1660537522453]
s1,t1 = polyfit(X, Cu100, 1) 
print( 'slope %2.3f ' %s1)
print( 'intersect %2.3f ' %t1)
x_1 = np.arange(0, 3/9, 0.01)

ax.annotate('Cu(100)', xy=(0.38,205), xytext=(-60,5), size= text_size, textcoords='offset points', color=colors2[10])
#plt.plot(x_1, s1*x_1 ,color=colors2[10],ls=':')

# Interpolation
f = interp1d(X, Cu100, kind='cubic')  # Use cubic spline interpolation

# Generate smoothly connected curve
x_smooth = np.linspace(min(X), max(X), 100)
y_smooth_Cu100 = f(x_smooth)
# Plotting
plt.plot(x_smooth, y_smooth_Cu100, color=colors2[10], ls=':')



ax.annotate('NiO(100)', xy=(0.375,230), xytext=(-65,-15), size= text_size, textcoords='offset points', color=datacolor[1])
NiO100 = [Gamma_NiO100[0]*1000, Gamma_NiO100_CO[0]*1000, Gamma_NiO100_2CO[0]*1000, Gamma_NiO100_3CO[0]*1000]
s2,t2 = polyfit(X, NiO100, 1) 
print( 'slope %2.3f ' %s2)
print( 'intersect %2.3f ' %t2)
#plt.plot(x_1, s2*x_1 ,color=datacolor[1],ls=':')
f_NiO = interp1d(X, NiO100, kind='cubic')  # Use cubic spline interpolation

# Generate smoothly connected curve
y_smooth_NiO100 = f_NiO(x_smooth)
# Plotting
plt.plot(x_smooth, y_smooth_NiO100, color=datacolor[1])



ax.annotate('Ni(100)', xy=(0.37,175), xytext=(-45,15), size= text_size, textcoords='offset points', color=datacolor[1])
Ni100 = [Gamma_Ni100[0]*1000, Gamma_Ni100_CO[0]*1000, Gamma_Ni100_2CO[0]*1000, Gamma_Ni100_3CO[0]*1000]
s0,t0 = polyfit(X, Ni100, 1) 
print( 'slope %2.3f ' %s0)
print( 'intersect %2.3f ' %t0)
plt.plot(x_1, s0*x_1+t0 ,color=datacolor[1],ls=':')



ax.annotate('1MLCu', xy=(0.4,280), xytext=(-70,-5), size= text_size, textcoords='offset points', color=colors2[3])

CuNiO100_1ML = [Gamma_1MLCuNiO100[0]*1000, Gamma_1MLCuNiO100_CO[0]*1000, Gamma_1MLCuNiO100_2CO[0]*1000, Gamma_1MLCuNiO100_3CO[0]*1000]
s3,t3 = polyfit(X, CuNiO100_1ML, 1) 
print( 'slope %2.3f ' %s3)
print( 'intersect %2.3f ' %t3)
#plt.plot(x_1, s3*x_1 ,color=colors2[3],ls=':')
f_CuNiO100_1ML = interp1d(X, CuNiO100_1ML, kind='cubic')  # Use cubic spline interpolation

# Generate smoothly connected curve
y_smooth_CuNiO100_1ML = f_CuNiO100_1ML(x_smooth)
# Plotting
plt.plot(x_smooth, y_smooth_CuNiO100_1ML, color=colors2[3], ls=':')



ax.annotate('2MLCu', xy=(0.4,320), xytext=(-70,-5), size= text_size, textcoords='offset points', color=colors2[4])

CuNiO100_2ML = [Gamma_2MLCuNiO100[0]*1000, Gamma_2MLCuNiO100_CO[0]*1000, Gamma_2MLCuNiO100_2CO[0]*1000, Gamma_2MLCuNiO100_3CO[0]*1000]
s4,t4 = polyfit(X, CuNiO100_2ML, 1) 
print( 'slope %2.3f ' %s4)
print( 'intersect %2.3f ' %t4)
#plt.plot(x_1, s4*x_1 ,color=colors2[4],ls=':')

f_CuNiO100_2ML = interp1d(X, CuNiO100_2ML, kind='cubic')  # Use cubic spline interpolation

# Generate smoothly connected curve
y_smooth_CuNiO100_2ML = f_CuNiO100_2ML(x_smooth)
# Plotting
plt.plot(x_smooth, y_smooth_CuNiO100_2ML, color=colors2[4], ls=':')


ax.annotate('3MLCu', xy=(0.4,345), xytext=(-70,-2), size= text_size, textcoords='offset points', color=colors2[6])

CuNiO100_3ML = [Gamma_3MLCuNiO100[0]*1000, Gamma_3MLCuNiO100_CO[0]*1000, Gamma_3MLCuNiO100_2CO[0]*1000, Gamma_3MLCuNiO100_3CO[0]*1000]
s5,t5 = polyfit(X, CuNiO100_3ML, 1) 
print( 'slope %2.3f ' %s5)
print( 'intersect %2.3f ' %t5)
#plt.plot(x_1, s5*x_1 ,color=colors2[6],ls=':')
f_CuNiO100_3ML = interp1d(X, CuNiO100_3ML, kind='cubic')  # Use cubic spline interpolation

# Generate smoothly connected curve
y_smooth_CuNiO100_3ML = f_CuNiO100_3ML(x_smooth)
# Plotting
plt.plot(x_smooth, y_smooth_CuNiO100_3ML, color=colors2[6], ls=':')

ax.annotate('4MLCu', xy=(0.4,360), xytext=(-70,-2), size= text_size, textcoords='offset points', color=colors2[8])
CuNiO100_4ML = [Gamma_4MLCuNiO100[0]*1000, Gamma_4MLCuNiO100_CO[0]*1000, Gamma_4MLCuNiO100_2CO[0]*1000, Gamma_4MLCuNiO100_3CO[0]*1000]
s6,t6 = polyfit(X, CuNiO100_4ML, 1) 
print( 'slope %2.3f ' %s6)
print( 'intersect %2.3f ' %t6)
#plt.plot(x_1, s6*x_1 ,color=colors2[8],ls=':')

f_CuNiO100_4ML = interp1d(X, CuNiO100_4ML, kind='cubic')  # Use cubic spline interpolation

# Generate smoothly connected curve
y_smooth_CuNiO100_4ML = f_CuNiO100_4ML(x_smooth)
# Plotting
plt.plot(x_smooth, y_smooth_CuNiO100_4ML, color=colors2[8], ls=':')


ax.set_xlabel(r'$\theta _{CO}$',fontsize=36)
ax.set_ylabel(r'$\gamma _{\rm surface} $ [meV/$\rm \AA ^2$]',fontsize=36)

#ax.legend(loc=1,bbox_to_anchor = [1.25, 0.95],fontsize=11)
#ax.legend(loc=3,fontsize=24)                                                                            
ax.set_xlim([-0.02,0.4])
#ax.xaxis.set_ticks([0, 1, 2, 3])
labels = [item.get_text() for item in ax.get_xticklabels()]
labels[0] = '0'
labels[1] = '1/9'
labels[2] = '2/9'
labels[3] = '3/9'
#ax.set_xticklabels(labels)

plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
ymin=160
ymax=380
ax.set_ylim([ymin,ymax])
ax.yaxis.set_ticks(np.arange(ymin,ymax+1,50))
ax.yaxis.set_minor_locator(MultipleLocator(5))
ax.xaxis.set_major_locator(MultipleLocator(0.1))
#fig.savefig('plot_ORR.png',bbox_inches='tight') 


bwidth =2
ax.spines['bottom'].set_linewidth(bwidth)
ax.spines['left'].set_linewidth(bwidth)
ax.spines['top'].set_linewidth(bwidth)
ax.spines['right'].set_linewidth(bwidth)
fig.savefig('Surface_free_energy_absolute_NiO_Cu_vs_Ni.png',bbox_inches='tight',dpi=300)
