import numpy as np
from scipy.stats import nct as ncttest
from scipy.stats import ncf as ncftest

def sprtBoundsAB(alpha,beta):
    return (np.log((1 - beta) / alpha),np.log(beta / (1 - alpha)))

def sprtTTestBoundsAB(alpha,beta):
    return ((1 - beta) / alpha,beta / (1 - alpha))


# Define function to calculate log likelihood ratio
def sprt_log_likelihood_ratio(x, mu_benchmark, sigma_benchmark, sigma_algorithm):
    return (-(x - mu_benchmark)**2 / (2 * sigma_benchmark**2) +
            (x - mu_benchmark)**2 / (2 * sigma_algorithm**2))

def checkHypothesisThatNewMeanIsLessThanComparison(oldS,newsample, mu_comparison, sigma_comparison, sigma_new,alpha,beta):
        S =oldS + sprt_log_likelihood_ratio(newsample, mu_comparison, sigma_comparison, sigma_new)
        A,B=sprtBoundsAB(alpha,beta)    
        if S >= A:
            #"New sample mean < previous mean"
            return (1,S)
        if S <= B:
            #"New sample mean >= previous mean"
            return (0,S) 
        
        return (-1,S)

# Sequential T-test

def pooledStandardDeviation(samples1,samples2):
    n1=len(samples1);n2=len(samples2)
    if n1< 2 or n2 < 2:
         print(f'invalid sample sizes: {n1} and {n2}')
         return None
    return np.sqrt(
                    ((n1-1)*np.var(samples1) + (n2-1)*np.var(samples2))
                    /
                    (n1+n2-2)
         )

def t(samples1,samples2):
    n1=len(samples1);n2=len(samples2)
    if n1< 2 or n2 < 2:
         print(f'invalid sample sizes: {n1} and {n2}')
         return None
    return (np.mean(samples1) - np.mean(samples2))/(pooledStandardDeviation(samples1,samples2)*(np.sqrt(1/n1 + 1/n2)))

# CohensD= (mean1-mean2)/sigma
# -> CohensD > 0 -> test if mean1 > mean2
def oneSidedTRatio(samples1,samples2,cohensD):
    n1=len(samples1);n2=len(samples2)
    delta=cohensD*np.sqrt((n1*n2)/(n1 + n2))
    tstat=t(samples1,samples2)
    df=n1+n2-2
    return ncttest.pdf(tstat,df=df,nc=delta)/ncttest.pdf(tstat,df=df,nc=0)

# CohensD= (mean1-mean2)/sigma
def twoSidedTRatio(samples1,samples2,cohensD):
    n1=len(samples1);n2=len(samples2)
    delta=cohensD*np.sqrt((n1*n2)/(n1 + n2))
    tstat=t(samples1,samples2)
    tstatsquared=tstat*tstat
    df=n1+n2-2
    if ncftest.pdf(tstatsquared,1,df,nc=delta*delta) == 0:
        print('zero R')
    return ncftest.pdf(tstatsquared,1,df,nc=delta*delta)/ncftest.pdf(tstatsquared,1,df,nc=0)

def checkIfNewMeanIsLessThanComparisonSequentialT(samples1,samples2,alpha,beta,cohensD):
    A,B=sprtTTestBoundsAB(alpha,beta)    
    R=oneSidedTRatio(samples1,samples2,cohensD)
    print(f"B,A = {B}-{A} R : {R} ")
    sigma=pooledStandardDeviation(samples1,samples2)
    if R >= A:
        #"New sample mean < previous mean"
        return 1
    if R <= B:
        #"New sample mean >= previous mean"
        return 0
        
    return -1

def checkIfNewMeanIsDifferentThanComparisonSequentialT(samples1,samples2,alpha,beta,cohensD):
    A,B=sprtTTestBoundsAB(alpha,beta)    
    R=twoSidedTRatio(samples1,samples2,cohensD)
    print(f"B,A = {B}-{A} R : {R} ")
    sigma=pooledStandardDeviation(samples1,samples2)
    if R >= A:
        #"New sample mean < previous mean"
        return 1
    if R <= B:
        #"New sample mean >= previous mean"
        return 0
        
    return -1