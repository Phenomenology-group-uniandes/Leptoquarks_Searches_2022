import ROOT
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy import signal as sig
from scipy.interpolate import splev, splrep
from scipy import interpolate
from sklearn.model_selection import GridSearchCV
from ROOT import *
from xgboost import XGBRegressor
DEF_PARAMETERS = {
    "n_estimators":[
        #10,
        #25,
        #50,
        75,
        #100,
        #250,
    ],
    "max_depth":[
        #1,
        3,
        #5,
        #7,
        #9
    ],
    "learning_rate":[
        #0.01,
        0.1,
        #1,
        #10,
        #100
    ],
    "eta":[
        0.1
    ],
    "subsample":[0.7],
    "colsample_bytree":[0.8],
    "monotone_constraints":["(1)"],
    "nthread":[4]
}

def getall(d, basepath="/"):
    "Generator function to recurse into a ROOT file/dir and yield (path, obj) pairs"
    for key in d.GetListOfKeys():
        kname = key.GetName()
        if key.IsFolder():
            # TODO: -> "yield from" in Py3
            for i in getall(d.Get(kname), basepath+kname+"/"):
                yield i
        else:
            yield d.Get(kname)
            


def butter_filtfilt( x, Wn=0.5, axis=0 ):
    """ butter( 2, Wn ), filtfilt
        axis 0 each col, -1 each row
    """
    b, a = sig.butter( N=2, Wn=Wn )
    return sig.filtfilt( b, a, x, axis=axis, method="gust" )  # twice, forward backward

def ints( x ):
    return x.round().astype(int)

def minavmax( x ):
    return "min av max %.3g %.3g %.3g" % (
        x.min(), 
        x.mean(), 
        x.max() 
    )

def pvec( x ):
    n = len(x) // 25 * 25
    return "%s \n%s \n" % (
        minavmax( x ),
        ints( x[ - n : ]) .reshape( -1, 25 ))

def monofit( a, Wn):
    y=a
    """ monotone-increasing curve fit """
    y = np.asarray(y).squeeze() 
    ygrad = np.gradient( y )
    gradsmooth = butter_filtfilt( ygrad, Wn=Wn )  
    ge0 = np.fmax( gradsmooth, 1e-4 )
    ymono = np.cumsum( ge0 )  # integrate, sensitive to first few
    ymono += (y - ymono).mean()
    err = y - ymono
    errstr = "average |y - monofit|: %.2g" % np.abs( err*100 ).mean()
    ymono=ymono
    return ymono, err, errstr



def interpolate_data(x,y,step,pot):
    interp_func_mid=PchipInterpolator(x, y)    
    X=np.power(np.arange(0, np.exp(0), step)+step,pot)
    Y=interp_func_mid(X)
    return X,Y

def smooth_interpolation_data(Y,step):
    Wn=step*100
    ymono, err, errstr = monofit( Y , Wn )
    Y=ymono/max(ymono)
    return Y

def get_data_by_histo(h):
    histo=h.Clone()
    x=[]
    y=[]
    err_y=[]
    n_events=histo.Integral()
    histo.Scale(1.0/n_events)
    sum_=0.
    erry=0.0
    dx=1./histo.GetNbinsX()
    for i in range (histo.GetNbinsX()):
        if histo.GetBinContent(i+1) !=0 :
            x.append((i+1)*(dx))
            sum_+=histo.GetBinContent(i+1)
            erry+=h.GetBinError(i+1)
            y.append(sum_)
            err_y.append(erry)
    #if x[len(x)-1]<1 :
    #    x.append(1.)
    #    y.append(1.)
    #    err_y.append(err_y[len(err_y)-1])
    x=np.array(x)
    y=np.array(y)

    err_y=np.array(err_y)
    
    
    y_sup=y+err_y
    y_inf=y-err_y

    y_sup=y_sup#/max(y_sup)
    y_inf=y_inf#/max(y_inf)
    
    interp_func_sup=PchipInterpolator(x, y_sup)
    interp_func_inf=PchipInterpolator(x, y_inf)
    
    interp_func = PchipInterpolator(x, y)
    new_x = np.arange(0.0, 1.00, 1e-6)
    new_y_sup = interp_func_sup(new_x)
    new_y_inf = interp_func_inf(new_x)
    
    X=new_x
    Y=[np.random.normal((b+a)/2, (b-a)/8+1e-8) for a,b in zip(new_y_inf,new_y_sup)]

    X=np.array(X)
    Y=np.array(Y)
    
    gbc = XGBRegressor()
    cv = GridSearchCV(
            gbc,
            DEF_PARAMETERS,
            n_jobs = 5,
            cv=2,
            verbose=True
        )
    xgb_x=np.vstack(X)
    xgb_y=Y
    cv.fit(xgb_x, xgb_y)
    print("the best parameters are", cv.best_params_)
    learning_rate=cv.best_params_["learning_rate"]
    n_estimators=cv.best_params_["n_estimators"]
    max_depth=cv.best_params_["max_depth"]
    model = XGBRegressor(
        n_estimators=n_estimators, 
        learning_rate=learning_rate,
        max_depth=max_depth, 
        eta=0.1, 
        subsample=0.7, 
        colsample_bytree=0.8,
        monotone_constraints="(1)",
        nthread=10
    )
    model.fit(xgb_x, xgb_y)
    
    yhat = model.predict(xgb_x)
    yhat = np.array([float(a) for a in yhat])
    yhat = yhat/max(yhat)
    
    return X , yhat , n_events





def get_bins_from_histo(h):
    bins=[h.GetBinContent(bin_+1) for bin_ in range(h.GetNbinsX())]
    simplify=[0.]
    for i, bin_ in enumerate(bins):
        if i < len(bins)/2:
            simplify[0]+=bin_
        else : 
            simplify+=[bin_]
            
    return h.GetName() ,  simplify


def refine_histo(h,n_bins=100):
    
    hist_test=TH1F(h.GetName(),h.GetTitle(),n_bins,0.,1.)
    hist_test.SetDirectory(0)
    interp_func = get_interp_func(h)
    for bin_ in range(hist_test.GetNbinsX()):
        hist_test.SetBinContent(bin_+1,interp_func((bin_+1)/n_bins)-interp_func((bin_)/n_bins))
    hist_test.Scale(h.Integral()/hist_test.Integral())
    return hist_test

def get_interp_func(h,step=1e-3,pot=2.4):
    x , y , n_events = get_data_by_histo(h)
    X , Y = interpolate_data(x,y,step,pot)
    Y=smooth_interpolation_data(Y,step)
    spl = interpolate.splrep(x, y, s=1e-3)
    step = step/5
    xnew = np.arange(0., np.exp(0), step)+step
    ynew = splev(xnew, spl)
    ynew=ynew/max(ynew)
    ynew = ynew*n_events
    return PchipInterpolator(xnew, ynew)

def sum_histos(histo_list):
    result=TH1F("sum","sum",histo_list[0].GetNbinsX(),0.,1.)
    result.SetDirectory(0)
    for histo in histo_list:
        for i in range (histo.GetNbinsX()):
            sum_=result.GetBinContent(i+1)
            sum_+=histo.GetBinContent(i+1)
            result.SetBinContent(i+1,sum_)
            err_=result.GetBinError(i+1)
            err_+=histo.GetBinError(i+1)
            result.SetBinError(i+1,err_)
    return result

def get_all_dict(file_path):
    hist_dict={}
    f=TFile(file_path)
    for hh in getall(f):
        hh.SetDirectory(0)
        hist_dict.update({hh.GetName():hh})
    return hist_dict

def combine_histos(hist_dict,group_dict,refine_names):
    n_bins=100
    final_histos=[]
    for signal in group_dict.keys():
        hist_list=[]
        for keyword in group_dict[signal]:
            for key in hist_dict.keys():
                if keyword in key:
                    hist_list.append(hist_dict[key])
        h=sum_histos(hist_list)
        n_events=h.Integral()
        h.SetName(signal)
        h.SetTitle(signal+";ML-score;nevents(137/fb)")
        if h.GetName() in refine_names:
            h=refine_histo(h,n_bins)
        h.Rebin(int(h.GetNbinsX()/n_bins))
        final_histos+=[h]
    return final_histos


import os
def conver_histos_to_txt(hist_list,save_dir):
    def write_txt(key):
        path=os.path.join(save_dir,f'{key}.txt')
        with open(path, 'w') as f:
            [f.write(str(bin_)+"\n") for bin_ in bins[key]]
        return path
    bins=dict(map(get_bins_from_histo,hist_list))
    return list(map(write_txt,bins.keys()))

from ROOT import TCanvas
def draw_hist(h,save_dir):
    c1=TCanvas( f'c-{h.GetName()}', '', 0, 0, 1280, 720)
    c1.SetGrid()
    c1.SetLogy()
    h.Draw("HIST")
    c1.SaveAs(os.path.join(save_dir,f"{h.GetName()}.png"))
    del c1