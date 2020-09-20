
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from tqdm.notebook import tqdm
from scipy.interpolate import griddata
from scipy.interpolate import interp1d

def bs_call(S, K, T, r, vol):
    d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)


def bs_put(S, K, T, r, vol):
    d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return -S * norm.cdf(-d1) + np.exp(-r * T) * K * norm.cdf(-d2)


def bs_vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def bs_delta(S,K,T,r, q, sigma, call_put):
    d1 = (np.log(S / K) + (r - q) * T) / (sigma * np.sqrt(T)) + 0.5 * sigma * np.sqrt(T)
    return call_put*np.exp(-q*T) * norm.cdf( call_put * d1 )


def implied_vol(target_value, S, K, r, T, call_put, *args):
    max_iter = 500
    tol = 1.0e-5
    sigma = 0.5 #np.sqrt(2*np.pi/T * target_value / S) # brenner subrahmyan vol
    for i in range(0, max_iter):
        if call_put == 1:
            price = bs_call(S, K, T, r, sigma)
        else:
            price = bs_put(S, K, T, r, sigma)
        vega = bs_vega(S, K, T, r, sigma)
        diff = target_value - price  # our root
        if (abs(diff) < tol):
            return sigma
        sigma = sigma + diff/vega # https://quant.stackexchange.com/questions/7761/a-simple-formula-for-calculating-implied-volatility
    return sigma


def parametrisation( p, x, y, w ):
    return np.sum( w * (y - p[0] * x - p[1] * x**2 - p[2] )**2 )


def delta_obj(p, params ):
    d, q, T, a, b, c, call_put = params
    return -call_put*p/(np.sqrt(a*p+b*p**2+c))+call_put*0.5*np.sqrt(a*p+b*p**2+c) - norm.ppf( d / (call_put*np.exp(-q*T)) )


def vol_obj( p, params ):
    pri, S, K, r, T, call_put = params

    d1 = (np.log(S/K) + (r + 0.5*p**2)*T) / (p*np.sqrt(T))
    d2 = d1 - p * np.sqrt(T)

    return call_put * (S * norm.cdf(call_put*d1) - np.exp(-r * T) * K * norm.cdf(call_put*d2)) - pri


def solve_obj(obj, bounds, params, tol = 1e-5, verbose = True):
    lower_bound, upper_bound = bounds
    k, multiplier, counter, i0, loss, loss0, max_iter = 3, 2, 0, 0, 10, 10, 10
    first = True
    while abs(loss) >= tol:
        for i in np.linspace(lower_bound,upper_bound,k):
            loss = obj(i, params)
            if first:
                loss0 = loss
                first = False
            if loss0 > 0 and loss < 0:
                lower_bound, upper_bound = i0, i
                break
            loss0, i0 = loss, i
        k *= multiplier
        counter += 1
        if counter >= max_iter:
            if verbose:
                print(i, 'failed with', loss)
            return np.nan
    return i



def calc_iv( opts, underlying, call_put ):
    ivs = []
    temp = opts.copy()
    for idx in tqdm(temp.index):
        target_value = temp.price.loc[idx]
        K = temp.strike.loc[idx]
        S = underlying.close[-1]
        r = 0.0
        T = temp.time_to_expiration.loc[idx]
        bounds = (0,100)
        params = (target_value, S, K, r, T, call_put)
        x = implied_vol( target_value, S, K, r, T, call_put )
        if x != x:
            x = solve_obj( vol_obj, bounds, params )
            if x != x:
                print(idx, T, target_value, K, S)
        ivs.append(x)
    return ivs


def calc_delta(opts, underlying, call_put):
    opts['delta'] = np.nan

    for idx in opts.index:
        call_put = 1
        K = opts.strike.loc[idx]
        S = underlying.close[-1]
        r = 0.0
        q = 0.0
        T = opts.time_to_expiration.loc[idx]
        sigma = opts.implied_volatility.loc[idx]
        opts.loc[idx, 'delta'] = bs_delta(S, K, T, r, q, sigma, call_put)
    return opts


def delta_surface(options, call_put_options, underlying, call_put, verbose = False ):
    parameters = {}
    delta_ivs = {}
    expirations = np.unique(call_put_options.expiration)
    for ex in tqdm(expirations):
        all_options = options[(options.expiration == ex)].copy()
        opts = call_put_options[(call_put_options.expiration == ex)].copy()

        opts = opts.sort_values('log_moneyness')
        all_options = all_options.sort_values('log_moneyness')
        opts.reset_index(drop = True, inplace = True)
        all_options.dropna(inplace = True)
        all_options.reset_index(drop = True, inplace = True)
        delta_K = (all_options.strike - all_options.strike.shift(1)).values

        T = opts.time_to_expiration.iloc[0]
        y = (T * all_options.implied_volatility**2).values
        x = (np.log(all_options.strike / underlying.close[-1])).values
        weights = delta_K / np.sqrt(2*np.pi*y) * np.exp(-0.5*(x/np.sqrt(y)+0.5*np.sqrt(y))**2)
        p0 = np.array([0,0,0])
        # 1 because first value is nan, could make it zero
        result = minimize(parametrisation, p0, args = (x[1:], y[1:], weights[1:]) )
        a,b,c = result.x

        temp = []
        if len(opts) > 5:
            if call_put == 1:
                bounds = (0,1)
            else:
                bounds = (-1,0)
            for d in np.linspace( call_put * 0.1, call_put * 0.5, 9 ):
                params = ( d, 0, T, a, b, c, call_put)
                x = solve_obj( delta_obj, bounds, params, tol = 1e-5, verbose = verbose )
                iv = a*x+b*x**2+c
                temp.append(iv)
            delta_ivs[ex] = temp
            parameters[ex] = {'a': a, 'b': b, 'c': c}
        elif len(opts) != 0:
            for d in np.linspace( call_put * 0.1, call_put * 0.5, 9 ):
                iv = c
                temp.append(iv)
            delta_ivs[ex] = temp
            parameters[ex] = {'a': 0, 'b': 0, 'c': c}

    if call_put == 1:
        idx = np.linspace( 0.1, 0.5, 9 )
    else:
        idx = 1 - abs(np.linspace( -0.1, -0.5, 9 ))
    return pd.DataFrame( delta_ivs, index = idx ), parameters


def shift_curve(calls, puts, ex):
    try:
        shift = puts[puts.expiration == ex].sort_values('strike').implied_volatility.ffill().iloc[-1] - calls[ calls.expiration == ex ].sort_values('strike').implied_volatility.bfill().iloc[0]
    except:
        shift = 0
    calls[calls.expiration == ex].sort_values('strike', inplace = True)
    puts[puts.expiration == ex].sort_values('strike', inplace = True)
    puts.loc[puts.expiration == ex, 'implied_volatility'] = puts[puts.expiration == ex].implied_volatility - shift / 2
    calls.loc[calls.expiration == ex, 'implied_volatility'] = calls[calls.expiration == ex].implied_volatility.bfill() + shift/2
    return calls, puts


def create_surface(X,Y,Z, method):
   XX,YY = np.meshgrid(np.linspace(min(X),max(X), len(X) ),np.linspace(min(Y),max(Y),len(Y) ))
   ZZ = griddata(np.array([X,Y]).T,np.array(Z),(XX,YY), method=method ) # fill_value = np.median(Z)
   return XX,YY,ZZ

def standardise_expiration(df, standard_exp):
    temp = pd.Series( df.columns.values - standard_exp )
    idx1, idx2 = max( temp[temp<0].index.values ), min( temp[temp>0].index.values )
    x1 = df.iloc[:,idx1].values
    x2 = df.iloc[:,idx2].values
    t1 = df.iloc[:,idx1].name
    t2 = df.iloc[:,idx2].name
    return np.array( [interp1d([ t1, t2 ], [ x1[i], x2[i] ]  )(standard_exp)  for i in range(len(x1) )] )
