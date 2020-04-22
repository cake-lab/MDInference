#!env python

import numpy as np
import random

import cutils

class ModelChooser(object):
    
    def __init__(self, models=[], gamma=1., **kwargs):
        self.models = models
        self.gamma = gamma
        self.algo = "mdinference"
        if "algo" in kwargs:
            self.algo = kwargs["algo"].lower()
        self.previous_time_budget = 0.0
    
    
    def getEligableModels_old(self, t_u, t_l):
        eligable_models = self.models
        #for model in self.models:
        #    print model
        eligable_models = filter( (lambda m: m.mu < t_l), eligable_models)
        eligable_models = filter( (lambda m: m.mu + self.gamma * m.sigma < t_u), eligable_models)
        if len(eligable_models) == 0:
            eligable_models = [ min(self.models, key=(lambda m: m.mu)) ]
        
        return eligable_models
        
    @staticmethod
    def pickModelsExplore(models):
        weights = []
        for model in models:
            weights.append(model.accuracy)
        weights = np.array(weights)
        weights = weights / sum(weights)
        
        model_idx = cutils.choose_one(weights, len(weights))
        model = models[model_idx]
        
        return model
    
    @staticmethod
    def getLimits(t_sla, t_transfer, t_infer, t_network):
        t_u = t_sla - (t_transfer + t_network) # should also make a difference between travel and transfer
        t_l = t_u - t_infer
        return t_u, t_l
    
    def runNaive(self, t_sla, t_transfer, t_network, t_infer):
            # Naive only takes into account the target SLA and chooses a model based on this information.
            eligable_models = filter( 
                                (lambda m: m.mu < t_sla),
                                self.models
                            )
            if len(eligable_models) == 0:
                eligable_models = [ min(self.models, key=(lambda m: m.mu)) ]
                
            model = max( 
                        eligable_models,
                        key=(lambda m: m.accuracy)
                    )
            
            return model
    
    def runExploration(self, t_sla, t_transfer, t_network, t_infer):
            # Exploration is based on the widely Exp3 algorithm ?? and acts by choosing a model from the full set of models M probabilistically based on their accuracy
            model = self.pickModelsExplore(self.models)
            return model
    
    def runBudget1(self, t_sla, t_transfer, t_network, t_infer):
            # . Budget II also calculates a time budget but chooses randomly from the set of all models that will respond within the time budget.
            t_u, t_l = ModelChooser.getLimits(t_sla, t_transfer, t_infer, t_network)
            eligable_models = self.getEligableModels(t_u, t_l)
            
            weights = np.ones(len(eligable_models)) / float(len(eligable_models))
            model_idx = cutils.choose_one(weights, len(weights))
            model = eligable_models[model_idx]
        
            #model = np.random.choice(eligable_models)
            return model
    
    def runBudget2(self, t_sla, t_transfer, t_network, t_infer):
            # . Budget II also calculates a time budget but chooses randomly from the set of all models that will respond within the time budget.
            t_u, t_l = ModelChooser.getLimits(t_sla, t_transfer, t_infer, t_network)
            eligable_models = self.getEligableModels(t_u, t_l)
            model = self.pickModelsExplore(eligable_models)
            
            return model
    
    
    
    @staticmethod
    def pickModelFromEligable(eligable_models):
        '''
        Given a set of models, selects the model that has the highest accuracy.  In the future this may also pick a model based on the Exp3 criteria.
        '''
        return max(eligable_models, key=(lambda m: m.accuracy))
    
    def runMDInference(self, t_sla, t_transfer, t_network=0, **kwargs):
        
        ##################################
        ## Step1: Calculate time Budget ##
        ##################################
        if "t_budget" in kwargs:
            t_budget = kwargs["t_budget"]
        else:
            if t_network == 0: # if t_network isn't given then we assume it is just the same as the transfer time
                t_network = t_transfer
            t_budget = t_sla - (t_transfer + t_network)
        self.previous_time_budget = t_budget
        
        #########################
        ## Step2: Calcuate M_E ##
        #########################
        M_E = filter( (lambda m: m.mu + self.gamma * m.sigma < t_budget), self.models)
        if len(M_E) == 0:
            M_E = [ min(self.models, key=(lambda m: m.mu)) ]
        
        ##################################
        ## Step3: Pick a model M in M_E ##
        ##################################
        model = self.pickModelFromEligable(M_E)
        
        return model
    
    
    def pickModel(self, t_sla, t_transfer=0, t_network=0, t_infer=0, **kwargs):
        
        model = None
        
        if self.algo == "naive":
            model = self.runNaive(t_sla, t_transfer, t_network, t_infer)
        
        elif self.algo == "exploration":
            model = self.runNaive(t_sla, t_transfer, t_network, t_infer)
            
        elif self.algo == "budget1":
            model = self.runBudget1(t_sla, t_transfer, t_network, t_infer)
            
        elif self.algo == "budget2":
            model = self.runBudget2(t_sla, t_transfer, t_network, t_infer)
            
        else:
            return self.runMDInference(t_sla, t_transfer, t_network, **kwargs)
        
        return model
        
        ###################################
    
    def getMaxDims(self, **kwargs):
        max_height = max([ m.getInputHeight() for m in self.models])
        max_width = max([ m.getInputWidth() for m in self.models])

        return (max_height, max_width)
        
