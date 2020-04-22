#!env python

import time

import numpy as np
import mdinference as md


class SimulationOptions(md.Options):
    min_sla = 0
    max_sla = 1000
    sla_step = 10
    simulation_steps = 10000
    
    networks = None
    models = None
    
    cycle_sigma = False
    sigma_step = 10
    
    def __init__(self, **kwargs):
        super(SimulationOptions, self).__init__(**kwargs)
        
        if "steps" in kwargs and kwargs["steps"] is not None:
            self.simulation_steps = kwargs["steps"]
            
        if kwargs["networks"] is not None:
            self.networks = kwargs["networks"]
        
        if kwargs["models"] is not None:
            self.models = kwargs["models"]
        
        self.min_sla = kwargs["min_sla"]
        self.max_sla = kwargs["max_sla"]
        self.sla_step = kwargs["sla_step"]
        
        if kwargs["cycle_sigma"]:
            self.cycle_sigma = True
        if kwargs["sigma_step"]:
            self.sigma_step = kwargs["sigma_step"]
        
    
    def getOptionsLine(self):
        options = ""
        if self.greedy:
            options += ".greedy"
        if self.no_probability:
            options += ".no_prob"
        if self.aggressive:
            options += ".aggressive"
        if self.safe:
            options += ".safe"
        if self.related_greedy:
            options += ".related_greedy"
        if self.pure_random:
            options += ".pure_random"
        if self.related_random:
            options += ".related_random"
        return options
    
    def getSLASteps(self):
        return range(self.min_sla, self.max_sla + self.sla_step, self.sla_step)
    
    def __str__(self):
        return "SimulationOptions"


class Network(object):
    def __init__(self, mu, sigma, name):
        self.mu = mu
        self.sigma = sigma
        self.name = name
    def sample(self):
        return getSample(self.mu, self.sigma)
    def __str__(self):
        return "Network(%s +/- %s)" % (self.mu, self.sigma)

class Request(object):
    def __init__(self, network, t_infer, t_sla, ondevice_accuracy): #, mu=NETWORK_AVG, sigma=NETWORK_STD):
        self.t_infer = t_infer
        self.t_sla = t_sla
        self.ondevice_accuracy = ondevice_accuracy
        self.t_transfer = network.sample()
        self.t_network = self.t_transfer
    def getLimits(self):
        t_u = self.t_sla - (2* self.travel_time) # should also make a difference between travel and transfer
        t_l = t_u - self.t_infer
        return t_u, t_l

      

def getNetworks(names=None):
    
    networks = {}
    
    networks["Home"] = Network(92.020537, 48.375776, "Home")
    networks["Home high-sigma"] = Network(92.020537, 92.0, "Home high-sigma")
    networks["Campus"] = Network(57.873475, 30.787465, "Campus")
    networks["LTE SIM"] = Network(364.66, 170.59, "LTE SIM")
    networks["Artificial"] = Network(50.0, 25.0, "Artificial")
    networks["Artificial2"] = Network(100.0, 50.0, "Artificial2")
    networks["Perfect"] = Network(50.0, 0.0, "Perfect")
    
    if names is not None:
        filtered_names = [n_name for n_name in networks.keys() if any([f_name.lower() in n_name.lower() for f_name in names])]
        print filtered_names
        if len(filtered_names) > 0:
            networks = { n_name : networks[n_name] for n_name in filtered_names}
        
    return networks
    

def getModelOracles_p2(names=None):
    modelOracles = {}
    
    modelOracles["densenet"]                        = md.models.ModelOracle( 64.2 , 25.4871516228, 0.14392328835)
    modelOracles["inception_resnet_v2"]             = md.models.ModelOracle( 77.5 , 50.8508994579, 0.330357097057)
    modelOracles["inception_v3"]                    = md.models.ModelOracle( 77.9 , 31.1075708866, 0.19144506006)
    modelOracles["inception_v4"]                    = md.models.ModelOracle( 80.1 , 59.2139463425, 0.220050309905)
    
    modelOracles["mobilenet_v1_0.25_128_frozen"]    = md.models.ModelOracle( 41.4 , 2.45355319977, 0.191013291683)
    modelOracles["mobilenet_v1_0.25_160_frozen"]    = md.models.ModelOracle( 45.4 , 3.14784097671, 0.0990715092523)
    modelOracles["mobilenet_v1_0.25_192_frozen"]    = md.models.ModelOracle( 47.1 , 3.16882705688, 0.077381567513)
    modelOracles["mobilenet_v1_0.25_224_frozen"]    = md.models.ModelOracle( 49.7 , 3.20725774765, 0.077706234051)
    
    modelOracles["mobilenet_v1_0.5_128_frozen"]     = md.models.ModelOracle( 56.2 , 3.65022063255, 0.112734896543)
    modelOracles["mobilenet_v1_0.5_160_frozen"]     = md.models.ModelOracle( 59.0 , 3.82574725151, 0.0655038470826)
    modelOracles["mobilenet_v1_0.5_192_frozen"]     = md.models.ModelOracle( 61.7 , 3.8555958271, 0.0653904943682)
    modelOracles["mobilenet_v1_0.5_224_frozen"]     = md.models.ModelOracle( 63.2 , 4.21054983139, 0.06411823069)
    
    modelOracles["mobilenet_v1_0.75_128_frozen"]    = md.models.ModelOracle( 62.0 , 3.66070413589, 0.111650728433)
    modelOracles["mobilenet_v1_0.75_160_frozen"]    = md.models.ModelOracle( 65.2 , 4.34910655022, 0.117551724545)
    modelOracles["mobilenet_v1_0.75_192_frozen"]    = md.models.ModelOracle( 67.1 , 4.39307403564, 0.0727340032091)
    modelOracles["mobilenet_v1_0.75_224_frozen"]    = md.models.ModelOracle( 68.3 , 4.6717915535, 0.0661689035893)
    
    modelOracles["mobilenet_v1_1.0_128_frozen"]     = md.models.ModelOracle( 65.2 , 3.82421827316, 0.0861939323294)
    modelOracles["mobilenet_v1_1.0_160_frozen"]     = md.models.ModelOracle( 68.0 , 4.17665553093, 0.101866251209)
    modelOracles["mobilenet_v1_1.0_192_frozen"]     = md.models.ModelOracle( 69.9 , 4.6239361763, 0.0823827471661)
    modelOracles["mobilenet_v1_1.0_224_frozen"]     = md.models.ModelOracle( 71.0 , 5.43101811409, 0.112308441232)
    
    modelOracles["nasnet_large"]                    = md.models.ModelOracle( 82.6 , 112.606483459, 0.335301277859)
    modelOracles["nasnet_mobile"]                   = md.models.ModelOracle( 73.9 , 21.1833128929, 0.166472799236)
    modelOracles["squeezenet"]                      = md.models.ModelOracle( 49.0 , 4.90813446045, 0.0579412476401)
    
    
    modelOracles["nasnet_pathos_acc"]                    = md.models.ModelOracle( 50.0 , 112.606483459, 0.335301277859)
    
    return modelOracles



def getModelOracles_c5(names=None):
    modelOracles = {}
    
    modelOracles["densenet"]                        = md.models.ModelOracle( 64.2 , 153.025884271, 29.6913487979)
    modelOracles["inception_resnet_v2"]             = md.models.ModelOracle( 77.5 , 500.985020161, 1.14565879578)
    modelOracles["inception_v3"]                    = md.models.ModelOracle( 77.9 , 262.045269012, 0.885511114865)
    modelOracles["inception_v4"]                    = md.models.ModelOracle( 80.1 , 540.745566368, 1.90960421806)
    
    modelOracles["mobilenet_v1_0.25_128_frozen"]    = md.models.ModelOracle( 41.4 , 3.95581150055, 0.201440857507)
    modelOracles["mobilenet_v1_0.25_160_frozen"]    = md.models.ModelOracle( 45.4 , 6.62793016434, 0.250722786275)
    modelOracles["mobilenet_v1_0.25_192_frozen"]    = md.models.ModelOracle( 47.1 , 8.76424479485, 0.397066544358)
    modelOracles["mobilenet_v1_0.25_224_frozen"]    = md.models.ModelOracle( 49.7 , 10.9814586639, 0.316590703247)
    
    modelOracles["mobilenet_v1_0.5_128_frozen"]     = md.models.ModelOracle( 56.2 , 8.26227641106, 0.200327773653)
    modelOracles["mobilenet_v1_0.5_160_frozen"]     = md.models.ModelOracle( 59.0 , 11.5909786224, 0.510917125336)
    modelOracles["mobilenet_v1_0.5_192_frozen"]     = md.models.ModelOracle( 61.7 , 15.0076212883, 0.273321275215)
    modelOracles["mobilenet_v1_0.5_224_frozen"]     = md.models.ModelOracle( 63.2 , 19.856048584, 0.251523968396)
    
    modelOracles["mobilenet_v1_0.75_128_frozen"]    = md.models.ModelOracle( 62.0 , 12.5075278282, 0.553963714457)
    modelOracles["mobilenet_v1_0.75_160_frozen"]    = md.models.ModelOracle( 65.2 , 17.3106851578, 0.357290436335)
    modelOracles["mobilenet_v1_0.75_192_frozen"]    = md.models.ModelOracle( 67.1 , 23.6952650547, 0.343032968511)
    modelOracles["mobilenet_v1_0.75_224_frozen"]    = md.models.ModelOracle( 68.3 , 31.5080704689, 0.242498263236)
    
    modelOracles["mobilenet_v1_1.0_128_frozen"]     = md.models.ModelOracle( 65.2 , 17.4309797287, 0.428864189945)
    modelOracles["mobilenet_v1_1.0_160_frozen"]     = md.models.ModelOracle( 68.0 , 25.2943475246, 0.37315250947)
    modelOracles["mobilenet_v1_1.0_192_frozen"]     = md.models.ModelOracle( 69.9 , 34.6836748123, 0.266217308913)
    modelOracles["mobilenet_v1_1.0_224_frozen"]     = md.models.ModelOracle( 71.0 , 46.1788566113, 0.220287887826)
    
    modelOracles["nasnet_large"]                    = md.models.ModelOracle( 82.6 , 1458.19308114, 6.66281445467)
    modelOracles["nasnet_mobile"]                   = md.models.ModelOracle( 73.9 , 100.973119497, 1.01353953097)
    modelOracles["squeezenet"]                      = md.models.ModelOracle( 49.0 , 38.0863963366, 11.2499465878)
    
    
    
    return modelOracles
    

def getModelOracles_old(names=None):
    modelOracles = {}
    modelOracles["MobileNet_v1_1.0_224"]    = md.models.ModelOracle(70.7, 32.93036599, 0.6586073198)
    modelOracles["MobileNet_v1_0.75_224"]   = md.models.ModelOracle(68.4, 20.03036748, 0.4006073495)
    modelOracles["MobileNet_v1_0.50_224"]   = md.models.ModelOracle(64, 10.2027503, 0.204055006)
    modelOracles["MobileNet_v1_0.25_224"]   = md.models.ModelOracle(50.6, 3.507560998, 0.07015121997)
    modelOracles["inception_v3"]            = md.models.ModelOracle(78.8, 43.9011, 0.8780)
    modelOracles["ResNet50"]                = md.models.ModelOracle(79.26, 123.0637553, 2.461275106)
    modelOracles["ResNet101"]               = md.models.ModelOracle(80.13, 197.6776103, 3.953552207)
    modelOracles["ResNet152"]               = md.models.ModelOracle(80.62, 233.7592444, 4.675184888)
    modelOracles["SqueezeNet"]              = md.models.ModelOracle(57.5, 0.05238100856, 0.001047620171)
    modelOracles["AlexNet"]                 = md.models.ModelOracle(57.2, 116.8094101, 2.336188202)
    modelOracles["VGG16"]                   = md.models.ModelOracle(75.3, 268.0754291, 5.361508581)
    
    
    return modelOracles


def getModelOracles(switch='p2', names=None, duplicate=1):
    
    modelOracles = {}
    
    for i in range(duplicate):
        if switch == 'c5':
            new_modelOracles = getModelOracles_c5()
        elif switch == 'old':
            new_modelOracles = getModelOracles_old()
        else:
            new_modelOracles = getModelOracles_p2()
        for key in new_modelOracles.keys():
            modelOracles[key + str(i)] = new_modelOracles[key]
    
    for model in modelOracles:
        modelOracles[model].name = model
    
    if names is not None:
        filtered_names = [n_name for n_name in modelOracles.keys() if any([f_name.lower() in n_name.lower() for f_name in names])]
        if len(filtered_names) > 0:
            modelOracles = { n_name : modelOracles[n_name] for n_name in filtered_names}
    
    return modelOracles



def getSample(mu, sigma):
    sample = np.random.normal(mu, sigma)
    while sample <= 0:
        sample = np.random.normal(mu, sigma)
    return sample


def runSimulation(network, model_chooser, sla=400, num_steps=1000, t_infer=000, ondevice_accuracy=0.5, **kwargs):
    accuracies = []
    infer_times = []
    pick_times = []
    names = []
    choosing_time = 0.
    for _ in range(num_steps):
        request = Request(network, t_infer, sla, ondevice_accuracy)
        
        start_time = time.time()
        model = model_chooser.pickModel(request.t_sla, request.t_transfer, request.t_network, request.t_infer, method="traditional", **kwargs)
        pick_time = 1000.0 * (time.time() - start_time)
        
        infer_time = getSample(model.mu, model.sigma) + request.t_transfer + request.t_network
        #   model.addInferenceTime(infer_time)
        
        infer_times.append(infer_time)
        pick_times.append(pick_time)
        accuracies.append(model.accuracy)
        names.append(model.name)
    return infer_times, accuracies, names, pick_times



def processNetworks(model_chooser, networks, options, append=False):
    
    options_line = options.getOptionsLine()
    #options_line = ""
    
    if not append:
        #with open("model_usage" + options + ".csv", 'w') as fid_usage:
        fid_usage = open("model_usage" + options_line + ".csv", 'w')
        fid_usage.write("network,goal_sla," + ','.join(sorted([m.name for m in model_chooser.models])) + ",algorithm" + "\n")
        
        #with open("results" + options + ".csv", 'w') as fid_sla:
        fid_sla = open("results" + options_line + ".csv", 'w')
        fid_sla.write("network,goal_sla,average_accuracy,average_time,num_samples,num_sla_violations,sla_error_rate,algorithm" + "\n")
        
        fid_times = open("times" + options_line + ".csv", 'w')
        fid_times.write("network,goal_sla,time,accuracy,model,algorithm,pick_time" + "\n")
    else:
        
        fid_usage = open("model_usage" + options_line + ".csv", 'a')
        fid_sla = open("results" + options_line + ".csv", 'a')
        fid_times = open("times" + options_line + ".csv", 'a')
        
    
    for network_name in sorted(networks.keys()):
        runProcessNetwork(model_chooser, networks[network_name], options.getSLASteps(), options, options_line, fid_usage, fid_sla, fid_times)
    

def runProcessNetwork(model_chooser, network, sla_steps, options, options_line, fid_usage, fid_sla, fid_times):
        for sla in sla_steps:
            
            # Run simulation step
            infer_times, accuracies, models, pick_times = runSimulation(network, model_chooser, sla=sla, num_steps=options.simulation_steps, options=options)
            
            print "SLA:", sla, "(%s) - (%s) - %0.2fms - %0.2f%%" % (network.name, model_chooser.algo, np.mean(infer_times), np.mean(accuracies))
            
            # Write out time information for each request
            for i in range(len(infer_times)):
                fid_times.write(','.join([str(t) for t in [network.name, sla, infer_times[i], accuracies[i], models[i], model_chooser.algo]]))
                fid_times.write(',' + str(pick_times[i]))
                fid_times.write('\n')
            
            # WRite out information for SLA step
            fid_sla.write(str( network.name ))
            fid_sla.write(",")
            
            fid_sla.write(str( sla ))
            fid_sla.write(",")
            
            fid_sla.write(str( np.mean(accuracies) ))
            fid_sla.write(",")
            
            fid_sla.write(str( np.mean(infer_times) ))
            fid_sla.write(",")
            
            fid_sla.write(str( options.simulation_steps ))
            fid_sla.write(",")
            
            fid_sla.write(str( len([s for s in infer_times if s > sla]) ))
            fid_sla.write(",")
            
            fid_sla.write(str( float(len([s for s in infer_times if s > sla])) / options.simulation_steps ))
            fid_sla.write(",")
            
            fid_sla.write(model_chooser.algo)
            fid_sla.write(",")
            
            fid_sla.write('\n')
            
            
            # Write out model usage information for SLA step
            fid_usage.write("%s,%s," % (network.name, sla))
            fid_usage.write(','.join([str(models.count(m)) for m in sorted([m.name for m in model_chooser.models])]) + (",%s" % model_chooser.algo) + "\n")
                    

