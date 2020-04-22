#!env python

import argparse

import numpy as np
import mdinference as md

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from Shared import getModelOracles, getNetworks, SimulationOptions, runSimulation, processNetworks







def main(**kwargs):
    
    options = SimulationOptions(**kwargs)
    
    
    modelOracles = getModelOracles(switch='p2') #, duplicate=100, names=["SqueezeNet"])
    networks = getNetworks(["Artificial2"])
    
    for i, algo in enumerate(["mdinference", "naive", "exploration", "budget1", "budget2"]):
        model_chooser = md.ModelChooser.ModelChooser(modelOracles.values(), algo=algo)
        processNetworks(model_chooser, networks, options, append=(i!=0))

    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--steps', '--num_steps', type=int, default=10000, help="Number of simluation steps")
    parser.add_argument('--greedy', action='store_true', help="Set to use greedy selection.")
    parser.add_argument('--no_probability', action='store_true', help="Disables the use of probabilitically choosing models")
    parser.add_argument('--aggressive', action='store_true', help="More aggressively chooses models")
    parser.add_argument('--safe', action='store_true', help="Plays it safer on the SLA")
    parser.add_argument('--networks', nargs='+', help="Names, or subnames, of networks to use")
    parser.add_argument('--models', nargs='+', help="Names, or subnames, of models to use")
    parser.add_argument('--min_sla', type=int, default=0)
    parser.add_argument('--max_sla', type=int, default=600)
    parser.add_argument('--sla_step', type=int, default=10)
    
    #parser.add_argument('--single_model', choices=["accuracy", "latency"], help="Set to use greedy.  Choices are latency or accuracy")
    
    parser.add_argument('--cycle_sigma', action='store_true', help="Runs through sigma values")
    parser.add_argument('--sigma_step', type=int, default=10)
    
    parser.add_argument('--related_greedy', action='store_true', help="Ignores sigma when finding a model")

    parser.add_argument('--pure_random', action='store_true', help="Picks from all models completely randomly")
    parser.add_argument('--related_random', action='store_true', help="Picks from the related models completely randomly")
    
    args = parser.parse_args()
    
    main(**vars(args))

