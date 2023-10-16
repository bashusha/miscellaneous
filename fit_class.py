
import traceback
import sys
import warnings
from typing import Union, Optional
import logging


import numpy as np
import pandas as pd
import scipy.stats as st

from scipy.stats._continuous_distns import _distn_names

#import statsmodels.api as sm

import matplotlib
import matplotlib.pyplot as plt

class FitDistribution():
    '''
    Find the best Probability Distribution Function for the given data
    '''
    
    def __init__(self,data,
                 distributionNames:list=None,
                 debug:bool=False,
                 censored:list=None,
                 metrics:list=None,
                 eps:float=1E-8
                ):
        '''
        constructor
        
        Args:
            data(dataFrame): the data to analyze
            distributionNames(list): list of distributionNames to try
            debug(bool): if True show debugging information
            censored(list): list of values of right bounds for censored data
        '''
        self.debug=debug
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        
        self.matplotLibParams()
        if distributionNames is None:
            self.distributionNames=[d for d in _distn_names if not d in ['levy_stable', 'studentized_range']]
        else:
            self.distributionNames=distributionNames
        
        if censored is None:
            self.censored = False
            self.data=data
            self.eps = eps
        else:
            self.censored = censored
            self.data = st.CensoredData.right_censored(data,  [(d in censored) for d in data]) # [(d in censored) for d in data] data in censored
            self.eps = -eps/2.
        # metrics used 
        self.metric_list_ = ['sse', 'r2', 'aic', 'bic', 'hqc', 'D', 'ks_p', 'chi_sq', 'chi_sq_p']
        if metrics:
            self.metrics = {m: np.nan for m in metrics if m in self.metric_list_}
            not_implemented_metrics = [m for m in metrics if not m in self.metric_list_]
            if not_implemented_metrics:
                warnings.warn(f'{str(not_implemented_metrics)} are not implemented' ) ##############
        else:
            self.metrics = {m: np.nan for m in self.metric_list_}
        logging.debug(self.metrics)    
        
    def matplotLibParams(self):
        '''
        set matplotlib parameters
        todo: set plotly parameters
        '''
        matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
        #matplotlib.style.use('ggplot')
        #matplotlib.use("WebAgg")

    # Create models from data
    def best_fit_distribution(self,bins:int=200, ax=None,density:bool=True):
        """
        Model data by finding best fit distribution to data
        """
        # Get histogram of original data
        y, x = np.histogram(self.data, bins=bins, density=density)
        x = (x + np.roll(x, -1))[:-1] / 2.0
    
        # Best holders
        best_distributions = []
        distributionCount=len(self.distributionNames)
        # Estimate distribution parameters from data
        for ii, distributionName in enumerate(self.distributionNames):
    
            print(f"{ii+1:>3} / {distributionCount:<3}: {distributionName}")
    
            distribution = getattr(st, distributionName)
    
            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    
                    # fit dist to data
                    params = distribution.fit(self.data)
    
                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]
                    
                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))
                    
                    # if axis pass in add to plot
                    try:
                        if ax:
                            pd.Series(pdf, x).plot(ax=ax)
                    except Exception:
                        pass
    
                    # identify if this distribution is better
                    best_distributions.append((distribution, params, sse))
            
            except Exception as ex:
                if self.debug:
                    trace=traceback.format_exc()
                    msg=f"fit for {distributionName} failed:{ex}\n{trace}"
                    print(msg,file=sys.stderr)
                pass
        
        return sorted(best_distributions, key=lambda x:x[2])

    def best_fit_distribution_df(self,*, bins:Union[int,list,str]='auto', 
                                 step:Optional[int],
                                 ax=None,density:bool=True): #bins:int=200
        """
        Model data by finding best fit distribution to data
        """
        # Get histogram of original data
        if step:
            data_min = self.data.min() if not self.censored else self.data._uncensor().min()
            data_max = self.data.max() if not self.censored else self.data._uncensor().max()
            bins_ = np.arange(data_min,data_max,step)            
            bins_ = np.append(bins_, data_max + self.eps)
        else:
            bins_ = bins # array-like or str
        
        if self.censored:
            y, x = np.histogram(self.data._uncensor(), bins=bins_, density=density)
        else:
            y, x = np.histogram(self.data, bins=bins_, density=density)
        
        if not step:
            bins_ = x
            bins[-1] += self.eps
        x = (x + np.roll(x, -1))[:-1] / 2.0
        n = len(self.data)
        logging.debug(f'num of bins: {len(bins_)}, bins: {bins_}, len(x): {len(x)}, x: {x}, len(y): {len(y)}')

        # contingency table
        value_cens = (self.data==data_max).sum() if not self.censored else (self.data._right==data_max).sum()
        if density:
            value_cens = value_cens * 1. / n
        ct = pd.DataFrame(data=np.append(y,value_cens), index=bins_, columns=['observed'])  #index=x[:-1]
        ct.loc[-np.inf,'observed'] = 0

        # Best holders
        best_distributions = []
        #distributionCount=len(self.distributionNames)
        
        # Estimate distribution parameters from data
        for distributionName in self.distributionNames:
    
            #print(f"{ii+1:>3} / {distributionCount:<3}: {distributionName}")
            print('.', end='')
    
            distribution = getattr(st, distributionName)
    
            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    #warnings.filterwarnings('ignore')
                    
                    # fit dist to data
                    params = distribution.fit(self.data)
    
                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]
                    
                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    dcdf = np.diff(distribution.cdf(bins_, loc=loc, scale=scale, *arg))
                    dcdf = np.append(dcdf, [1-distribution.cdf(data_max, loc=loc, scale=scale, *arg),distribution.cdf(data_min, loc=loc, scale=scale, *arg)])
                    if not density:
                        dcdf *= n   ########## elementwise
                    logging.debug(f'dcdf = {dcdf}, len = {len(dcdf)}')
                    ct[distributionName] = dcdf

                    sse = np.sum(np.power(y - pdf, 2.0)) 
                    if 'sse' in self.metrics:
                        self.metrics['sse'] = sse
                    if 'r2'  in self.metrics:
                        tse = (len(y)-1) * np.var(y, ddof=1)
                        self.metrics['r2'] = 1-sse/tse ##########
                    loglik = np.sum(distribution.logpdf(x, *params))
                    k = len(params[:])
                    
                    if 'aic' in self.metrics:
                        self.metrics['aic'] = 2 * k - 2 * loglik
                    if 'bic' in self.metrics:
                        self.metrics['bic'] = n * np.log(sse / n) + k * np.log(n)
                    if 'hqc' in self.metrics:
                        self.metrics['hqc'] = 2 * k * np.log(np.log(n)) - 2 * loglik  
                    if 'D' in self.metrics or 'ks_p' in self.metrics:
                        try:
                            self.metrics['D'], self.metrics['ks_p'] = st.kstest(self.data, distributionName, args=params)
                        except Exception:
                            pass
                    if 'chi_sq' in self.metrics or 'chi_sq_p' in self.metrics:
                        try:
                            self.metrics['chi_sq'], self.metrics['chi_sq_p'] = st.chisquare(np.append(y,[0,0]), dcdf)
                        except Exception:
                            pass



                    # if axis pass in add to plot
                    try:
                        if ax:
                            pd.Series(pdf, x).plot(ax=ax)
                    except Exception:
                        pass
    
                    
                    # identify if this distribution is better
                    best_distributions.append(
                        (distribution, params, arg, loc, scale, *self.metrics.values())) ###################### sse, r2, aic, bic, hqc, D, ks_p
            
            except Exception as ex:
                if self.debug:
                    trace=traceback.format_exc()
                    msg=f"fit for {distributionName} failed:{ex}\n{trace}"
                    print(msg,file=sys.stderr)
                pass
            return_df = pd.DataFrame(data=sorted(best_distributions, key=lambda x:x[5]),
                                     columns=['dist','params', 'arg', 'loc', 'scale'] + list(self.metrics.keys())) ########### 
            return_df['name'] = return_df['dist'].apply(lambda x: x.name)
            
        return return_df, ct.sort_index()

    def make_pdf(self,dist, params:list, size=10000):
        """
        Generate distributions's Probability Distribution Function 
        
        Args:
            dist: Distribution
            params(list): parameter
            size(int): size
            
        Returns:
            dataframe: Power Distribution Function 
        
        """
    
        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
    
        # Get sane start and end points of distribution
        start = dist.ppf(0.001, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.001, loc=loc, scale=scale)
        end   = dist.ppf(0.999, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.999, loc=loc, scale=scale)
    
        # Build PDF and turn into pandas Series
        x = np.linspace(start, end, size)
        y = dist.pdf(x, loc=loc, scale=scale, *arg)
        pdf = pd.Series(y, x)
    
        return pdf

    
        
    def analyze(self,title,x_label,y_label,*,
                outputFilePrefix=None,imageFormat:str='png',
                #allBins:str='auto',distBins:str='auto',
                allBins:int=500,distBins:int=20000,
                num_best:int=3,
                density:bool=True):
        """
        
        analyze the Probabilty Distribution Function
        
        Args:
            data: Panda Dataframe or numpy array
            title(str): the title to use
            x_label(str): the label for the x-axis
            y_label(str): the label for the y-axis
            outputFilePrefix(str): the prefix of the outputFile
            imageFormat(str): imageFormat e.g. png,svg
            allBins(int): the number of bins for all --> str 
                        method chosen to calculate the optimal bin width and consequently the number of bins
            distBins(int): the number of bins for the distribution --> same
            density(bool): if True show relative density
        """
        self.allBins=allBins
        self.distBins=distBins
        self.density=density
        self.title=title
        self.x_label=x_label
        self.y_label=y_label
        self.imageFormat=imageFormat
        self.outputFilePrefix=outputFilePrefix
        self.color=list(matplotlib.rcParams['axes.prop_cycle'])[1]['color']
        #self.best_dist=None
        #self.num_best=num_best
        self.analyzeAll()
        if outputFilePrefix is not None:
            self.saveFig(f"{outputFilePrefix}_{x_label}_All.{imageFormat}", imageFormat)
            plt.close(self.figAll)
        #display(self.best_distributions_df[['name','arg','loc','scale']+self.metric_list]) ############################
        
        if self.best_distributions_df.shape[0]: #self.best_dist: #.any():  # не было .any()
            num_best_adj = min(num_best,self.best_distributions_df.shape[0])
            for row in self.best_distributions_df.head(num_best_adj).itertuples(index=True): #():
                # self.pdf = self.make_pdf(self.best_dist['dist'], self.best_dist['params'])
                self.pdf = self.make_pdf(getattr(row, 'dist'), getattr(row, 'params'))
                self.analyzeBest(getattr(row, 'dist'), getattr(row, 'params'))
                if outputFilePrefix is not None:
                    print(getattr(row, 'Index'), end='; ')
                    self.saveFig(f"{outputFilePrefix}Best_{x_label}_{getattr(row, 'Index')}_{getattr(row, 'name')}.{imageFormat}", imageFormat)
                    plt.close(self.figBest)
            print('Done')
            
    def analyzeAll(self):
        '''
        analyze the given data
        
        '''
        # Plot for comparison
        figTitle=f"{self.title}\n All Fitted Distributions"
        self.figAll=plt.figure(figTitle,figsize=(12,8))
        ax = self.data.plot(kind='hist', bins=self.allBins, density=self.density, alpha=0.5, color=self.color)
        
        # Save plot limits
        dataYLim = ax.get_ylim()
        # Update plots
        ax.set_ylim(dataYLim)
        ax.set_title(figTitle)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        
        # Find best fit distribution
        self.best_distributions_df = self.best_fit_distribution_df(bins=self.distBins, ax=ax,density=self.density)
        
    def analyzeBest(self, dist, params):
        '''
        analyze the Best Property Distribution function
        '''
        # Display
        figLabel="PDF"
        self.figBest=plt.figure(figLabel,figsize=(12,8))
        ax = self.pdf.plot(lw=2, label=figLabel, legend=True)
        self.data.plot(kind='hist', bins=self.allBins, density=self.density, alpha=0.5, label='Data', legend=True, ax=ax,color=self.color)
        
        param_names = (dist.shapes + ', loc, scale').split(', ') if dist.shapes else ['loc', 'scale']
        param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, params)])
        dist_str = '{}({})'.format(dist.name, param_str)
        
        ax.set_title(f'{self.title} with best fit distribution \n' + dist_str)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)

    def saveFig(self,outputFile:str=None,imageFormat='png'):
        '''
        save the current Figure to the given outputFile
        
        Args:
            outputFile(str): the outputFile to save to
            imageFormat(str): the imageFormat to use e.g. png/svg
        '''
        plt.savefig(outputFile, format=imageFormat) # dpi 
     


