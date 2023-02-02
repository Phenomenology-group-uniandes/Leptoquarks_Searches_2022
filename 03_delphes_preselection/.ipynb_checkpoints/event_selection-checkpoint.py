import os
import time
import pandas as pd
import subprocess
from pathlib import Path
from ROOT import TChain, TLorentzVector
import subprocess
def add_parent_lib_path(name="Pheno_BSM"):
    import sys
    sys.path.append(
        os.path.join(
            sys.path[0].split(name)[0],
            name
        )
    )
add_parent_lib_path()
from delphes_reader import Quiet
from delphes_reader import DelphesLoader
from delphes_reader.clasificator import get_met
from delphes_reader.clasificator import get_jets
from delphes_reader.clasificator import get_good_jets
from delphes_reader.clasificator import get_good_leptons
from delphes_reader.clasificator import get_unified
from delphes_reader.root_analysis import get_kinematics_row

def get_sT(*args):
    particles=list(args)
    sT=0.0
    for particle in particles:
        sT+=particle.pt()
    return sT
def get_mT(*args):
    particles=list(args)
    sum_=TLorentzVector(0,0,0,0)
    for particle in particles:
        sum_+=particle.TLV
    return sum_.Mt()

#counter
def count_event(cut_dict,name):
    if not(cut_dict.get(name,None)):
        cut_dict.update({name:0})
    cut_dict[name]+=1
    return cut_dict

def b_sel(selection,b_jets,lep_a,lep_b):
    
    if (len(b_jets)==0):
        return None , None , None
    if (len(b_jets)==1):
        string=f"b_tau_tau_{selection}"
        condition="Exactly one b jet"
        b=b_jets[0]
        row=get_kinematics_row(b,lep_a,lep_b)
    else:
        string=f"b_b_tau_tau_{selection}"
        condition="At least two b jet"
        b_l=b_jets[0]
        b_s=b_jets[1]
        row=get_kinematics_row(b_l,b_s,lep_a,lep_b)
        
    Charges = lep_a.Charge*lep_b.Charge
    row.update(
        {f"Q_{{{lep_a.Name}}}Q_{{{lep_b.Name}}}":Charges}
    )
    return string, condition , row


def full_selection(tree,cutflow,results_dict):
    for event in tree:
        for key in results_dict.keys():
            count_event(cutflow[key],"All_events")
        jets = get_good_jets(event)
        l_jets=jets['l_jet']
        b_jets=jets['b_jet']
        tau_jets=jets['tau_jet']

        MET=get_met(event)
        row={
            "MET(GeV)":MET.pt(),
            "#phi_{MET}":MET.phi(),
            "light_jets_multiplicity": len(l_jets)
        }
        if (len(tau_jets)==0):
            ##Full-leptonic selection
            selection="leptonic"
            leptons=get_good_leptons(event)
            for key in results_dict.keys(): 
                if f"_{selection}" in key: 
                    count_event(
                        cutflow[key],
                        "Exactly zero good hadronic taus"
                    )
            if not( len(leptons) >= 2 ): continue
            for key in results_dict.keys(): 
                if f"_{selection}" in key: 
                    count_event(
                        cutflow[key],
                        "At Least two good leptons"
                    )
            lep_a=leptons[0]
            lep_b=leptons[1]
        elif (len(tau_jets)==1): 
            ##Semi-leptonic selection
            selection="semileptonic"
            leptons=get_good_leptons(event)

            for key in results_dict.keys(): 
                if f"_{selection}" in key: 
                    count_event(
                        cutflow[key],
                        "Exactly one good hadronic tau"
                    )
            if not( len(leptons) >=1 ): continue
            for key in results_dict.keys(): 
                if f"_{selection}" in key: 
                    count_event(
                        cutflow[key],
                        "At Least one good leptons"
                    )
            lep_a=tau_jets[0]
            lep_b=leptons[0]
        else:
            selection="hadronic"
            lep_a=tau_jets[0]
            lep_b=tau_jets[1]
            for key in results_dict.keys(): 
                if f"_{selection}" in key: 
                    count_event(
                        cutflow[key],
                        "At least two good hadronic taus"
                    )
        str_a, cond, kin_row = b_sel(selection, b_jets, lep_a, lep_b)
        if ( len(b_jets) == 1):
            kin_row["sT(GeV)"] = get_sT(b_jets[0],lep_a,lep_b,MET)
            kin_row["mT(GeV)"] = get_mT(b_jets[0],lep_a,lep_b,MET)
        elif(len(b_jets) == 2):
            kin_row["sT(GeV)"] = get_sT(b_jets[0],b_jets[1],lep_a,lep_b,MET)
        if (str_a):
            row.update(kin_row)
            count_event(cutflow[str_a],cond)
            results_dict[str_a]+=[row]

def eventSelection(signal,folder_out):
    start_time = time.time()

    #Get .root files list
    signal=DelphesLoader(signal)
    print("="*60, flush=True)
    nTrees=len(signal.Forest)
    cutflow={}
    
    #Declare dict to save results into csv files
    results_dict={
        #hadronic selection
        "b_tau_tau_hadronic":[],
        "b_b_tau_tau_hadronic":[],
        #semileptonic selection
        "b_tau_tau_semileptonic":[],
        "b_b_tau_tau_semileptonic":[],
        #leptonic selection
        "b_tau_tau_leptonic":[],
        "b_b_tau_tau_leptonic":[]
    }
    
    #Declare cut flow dict for each selection 
    for key in results_dict.keys():
        cutflow.update({key:{}})
        cutflow[key].update({"xs":signal.xs})
        
    
    for path_root in signal.Forest:
        #if i > 0 : break
        for attempt in range(10):
            try:
                cutflow_copy=cutflow.copy()
                results_dict_copy=results_dict.copy()
                tree=TChain("Delphes;1")
                tree.Add(path_root)
                full_selection(
                    tree,
                    cutflow_copy,
                    results_dict_copy
                )
            except RuntimeError as error:
                if not ("TTree I/O error" == error):
                    raise Exception("Something was wrong") 
                print("Reconnecting with Google drive")
                time.sleep(10)
                subprocess.call("bash ~/G_Drive_Recon.sh",shell=True)
                print("Okay, I'm going to try it next...")
            else:
                results_dict=results_dict_copy
                cutflow=cutflow_copy
                break
    for key in results_dict.keys():
        results_dict[key]=pd.DataFrame.from_records(results_dict[key])
        results_dict[key].to_csv(
            os.path.join(folder_out,f"{signal.name}_{key}.csv"), 
            index=False
        )

    elapsed=(time.time() - start_time)/3600.
    print(signal.name,"done!")
    print(f"time elapsed: {elapsed} hours.")
    print("%"*60, flush=True)
    return ( signal.name , cutflow )
    
def get_efficiencies_df(cutflow_dict,folder_out):
    dataframe_dict={}
    for name in next(iter(cutflow_dict.items()))[1].keys():
        #cut names
        results_dict={
            "cut":
            [a[0] for a in next(iter(cutflow_dict.items()))[1][name].items()]
        }
        results_dict["cut"].append("nEvents (137/fb)")
        for key in cutflow_dict.keys():
            signal_dict=cutflow_dict[key][name]
            if (len(results_dict["cut"]) != len(signal_dict)+1): continue
            results_dict.update({key:[]})
            last_cut=1.0
            for cut_key in signal_dict.keys():
                if "xs" == cut_key : 
                    results_dict[key].append(signal_dict[cut_key])
                else :
                    last_cut=signal_dict[cut_key]/signal_dict['All_events']
                    results_dict[key].append(last_cut)
            results_dict[key].append(last_cut*float(signal_dict["xs"])*137.*1000.)
            
        df=pd.DataFrame.from_dict(results_dict)
        df.to_csv(
            os.path.join(folder_out,f"Cutflow_{name}.csv"), 
            index=False
        )
        dataframe_dict.update(
            {name:df}
        )
    for selection in dataframe_dict.keys():
        df = dataframe_dict[selection]
    return dataframe_dict