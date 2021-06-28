from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from astropy.time import Time
import os, sys
from scipy.signal import lombscargle
from bokeh.palettes import viridis, inferno
import subprocess
from bokeh.plotting import output_notebook, figure, show
from bokeh.models import HoverTool, tools,ColumnDataSource, Whisker, ColorBar, LinearColorMapper
output_notebook()

from bokeh.models import LogColorMapper, LogTicker, ColorBar
from astropy.timeseries import LombScargle

import matplotlib.pyplot as plt
#%matplotlib widget
# use seaborn for plot styles
import seaborn; seaborn.set()




outputfile = 'outdir/out.photo'
columnfile = 'outdir/out.columns'




def getobjectdataold(datafile,index,name='name'):
    command = "sed '{}q;d'  {}".format(index+1,datafile)
    print(command)
    #datastring = ! {command}
    datalist = datastring[0].split()
    print(datalist[2:4])
    return datalist


def getobjectdata(datafile,index,name='name'):
    command = "sed '{}q;d'  {}".format(index+1,datafile)
    print(command)
    #datastring = ! {command}
    datastring2 = subprocess.check_output(command, shell=True)
    #print('a',datastring)
    datalist = datastring2.split()
    #print(datalist[2:4])
    return datalist





def getvegamaganddates(data,columnfile,name):
    vegamagindex = np.arange(28,6568,13)
    
    
    vegamaglist = np.array([float(data[i]) for i in vegamagindex])
    vegamaglisterrors = np.array([float(data[i+2]) for i in vegamagindex])
    
    photometricflag = np.array([float(data[i+8]) for i in vegamagindex])
    
    columns = []
    with open(columnfile) as filecol:
        for line in filecol:
            columns.append(line)
    listanames = []
    dates = []
    times =[]
    nameobject = []
    namel=[]

    for i in vegamagindex:
        initname = columns[i].find('Try/u')
        namefits = columns[i][initname:initname+23]
        listanames.append(namefits)
        fit = fits.open(namefits+'.fits')
        dates.append(fit[0].header['DATE-OBS'])
        times.append(fit[0].header['TIME-OBS'])
        namel.append(name)
    datesobs = [['{}T{}'.format(i,j)] for i,j in zip(dates,times)]

    d = np.array(datesobs)
    t = Time(d)
    namel = np.array(namel)
       
    return vegamaglist, vegamaglisterrors,t, d, namel, photometricflag


def createdatadiccomplete(data,columnfile):
    vegamag,vegamagerrros, timelist, observationname = getvegamaganddates(data,columnfile)
    datadic = {'time':timelist.datetime64,
               'VEGAMAG':vegamag,
               'obnames':observationname,
                'errorsmag':vegamagerrros}
    return datadic
            



def createdatadicclean(datafile,index,columnfile,name='name',timeformat='datetime64'):
    dataobject = getobjectdata(datafile,index)
    vegamag,vegamagerrros, timelist, observationname, namelist, photometricflag = getvegamaganddates(dataobject,columnfile,name)
    goodvegamagindex = np.where(vegamag < 99)[0]
    goodphotoindex = np.where(photometricflag == 0.)
    goodindex = np.intersect1d(goodvegamagindex,goodphotoindex)
    #print(goodindex)
    if timeformat == 'datetime64':
        timelist2 = timelist.datetime64[goodindex]
    elif timeformat == 'mjd':
        timelist2 = timelist.mjd[goodindex]

    datadic = {'time':np.array(timelist2.flatten()),
               'VEGAMAG':vegamag[goodindex],
               'obnames':observationname[goodindex],
                'errorsmag':vegamagerrros[goodindex],
               'upper':vegamag[goodindex] + vegamagerrros[goodindex] ,
              'lower':vegamag[goodindex] - vegamagerrros[goodindex],
              'name':namelist[goodindex]}
    return datadic


def plotlb(outputfile,index,columnfile,name='',plot=True,getarg=False,periods=np.linspace(0.2, 1.4, 4000)):
    datagoodmjd=  createdatadicclean(outputfile,index,columnfile,name=name,timeformat='mjd')
    
    
    t, mag, dmag = datagoodmjd['time'], datagoodmjd['VEGAMAG'],datagoodmjd['errorsmag']

        # Choose a period grid
    periods = periods
    #periods = np.linspace(0.2, 1.4, 4000)
    ang_freqs = 2 * np.pi / periods

    # compute the (unnormalized) periodogram
    # note pre-centering of y values!
    power = lombscargle(t.flatten(), mag - mag.mean(), ang_freqs,normalize=False,precenter=True)

    # normalize the power
    N = len(t.flatten())
    #power *= 2 / (N * mag.std() ** 2)
    if plot == True:
        # plot the results
        fig, ax = plt.subplots()
        ax.plot(periods, power)
        ax.set(ylim=(0, 0.8), xlabel='period (days)',
               ylabel='Lomb-Scargle Power');
    if getarg == True:
        return periods, power,t,mag,dmag

def plotlc(outputfile,indexlist,columnfile,namelist=''):
    colorlist = ['red','blue','green','yellow']
    p = figure(plot_width=900, plot_height=500, title='',active_drag='pan', active_scroll='wheel_zoom',
                  x_axis_type='datetime',y_axis_label='VEGAMAG',x_axis_label='Date_Obs')
    
    
    for i,indexnumber in enumerate(indexlist):
        color = colorlist[i]
        
        datagood = createdatadicclean(outputfile,indexnumber,columnfile,name=namelist[i])

        source = ColumnDataSource(data=datagood)
   


        


        #Tool to get wavelength
        hover2 = HoverTool(
                tooltips=[
                    ('Date', '(@obnames)')
                ]
            )


        p.add_tools(hover2)



        # add a circle renderer with a size, color, and alpha
        p.circle('time','VEGAMAG', source=source, color=color, name='name',legend='name')
        p.add_layout(
            Whisker(source=source, base="time", upper="upper", lower="lower")
        )


    p.y_range.flipped = True



    show(p)

def plotphase(outputfile,index,columnfile,name='',plot=False,periodmax=99.99, periods=np.linspace(0.2, 1.4, 4000),returnarg=False):
    nameob= name
    periods,power,t,mag,dmag = plotlb(outputfile,index,columnfile,name=nameob,plot=plot,getarg=True,periods=periods)
    #datagoodmjd=  createdatadicclean(outputfile,index,columnfile,name=name,timeformat='mjd')
    
    
    #t, mag, dmag = datagoodmjd['time'], datagoodmjd['VEGAMAG'],datagoodmjd['errorsmag']
    indexmax = np.where(power == power.max())
    #print(periods)
    if periodmax == 99.99:
        periodmax = periods[indexmax]
    print(periodmax)

    # Compute phases of the obsevations
    phase = (t.flatten() / periodmax) % 1

    # Plot the phased data & model
    fig, ax = plt.subplots()

    
    ax.errorbar(phase, mag, dmag, fmt='.k', ecolor='gray', alpha=0.5)
    ax.invert_yaxis();
    plt.show()
    if returnarg:
        return phase,mag,dmag, t.flatten()
    
    
def plotlcseaborn(outputfile,index,columnfile,name='',invertaxis = False):
    datagoodmjd=  createdatadicclean(outputfile,index,columnfile,name=name,timeformat='mjd')
    
    
    t, mag, dmag = datagoodmjd['time'], datagoodmjd['VEGAMAG'],datagoodmjd['errorsmag']

    
    
    fig, ax = plt.subplots()


    ax.errorbar(t, mag, dmag,  fmt='.k', ecolor='gray')
    ax.set(xlabel='Time (days)', ylabel='magitude',
           title=datagoodmjd['name'][0])
    if invertaxis == True:
        ax.invert_yaxis();
    plt.show()