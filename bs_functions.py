
import numpy as np
from scipy.stats import norm


def bs_value(S, K, T, r, q, sigma, call_put):
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return call_put * S * np.exp(-q*T) * norm.cdf(call_put * d1) - call_put * np.exp(-r * T) * K * norm.cdf( call_put * d2)


def bs_vega(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T) # /100 ?


def bs_delta(S, K, T, r, q, sigma, call_put):
    d1 = (np.log(S / K) + (r - q) * T) / (sigma * np.sqrt(T)) + 0.5 * sigma * np.sqrt(T)
    return call_put*np.exp(-q*T) * norm.cdf( call_put * d1 )


def bs_gamma(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q) * T) / (sigma * np.sqrt(T)) + 0.5 * sigma * np.sqrt(T)
    return np.exp(-q*T) / (S*sigma*np.sqrt(T)) * norm.pdf(d1)


def bs_theta(S, K, T, r, q, sigma, call_put):
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    x1 =  -(S*sigma*np.exp(-q*T)) / (2*np.sqrt(T)) * norm.pdf(d1)
    x2 = -call_put * r * K * np.exp(-r*T) * norm.cdf(call_put*d2) + call_put * q * S * np.exp(-q*T) * norm.cdf(call_put*d1)
    return (1/T) * (x1 + x2)


def bs_rho(S, K, T, r, q, sigma, call_put):
    d1 = (np.log(S / K) + (r - q) * T) / (sigma * np.sqrt(T)) + 0.5 * sigma * np.sqrt(T)
    d2 = d1 - sigma * np.sqrt(T)
    return call_put * K * T * np.exp(-r*T) * norm.cdf(call_put * d2)


def bs_vanna(S, K, T, r, q, sigma):
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    # vanna = vega * d2/(S*sigma)
    # return bs_vega(S, K, T, r, q, sigma) * d2/(S*sigma)
    return np.exp(-q*T) * np.sqrt(T) * norm.pdf(d1) * (d2/sigma)


def bs_volga(S, K, T, r, q, sigma):
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    # return bs_vega(S, K, T, r, q, sigma) * (d1*d2)/(S*sigma)
    return np.exp(-q*T) * np.sqrt(T) * norm.pdf(d1) * ((d1*d2)/sigma)


def bs_omega(S, K, T, r, q, sigma, call_put):
    return bs_delta(S, K, T, r, q, sigma, call_put) * S / bs_value(S, K, T, r, q, sigma, call_put)
