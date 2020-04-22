#!env python

import numpy as np
import os
import math
import scipy.stats
import time
import uuid

import logging

class AbstractModel(object):
    def __init__(self, name="", **kwargs): #, infer_avg=0.0, infer_std=0.0):
        self.name = name
        self.mu = 0.0
        self.sigma = 0.001
        self.accuracy = 1.0
        self.uuid = str(uuid.uuid4())
    def __str__(self):
        return "%s +- %s (%s%%)" % ( self.mu, self.sigma, self.accuracy )

class ModelOracle(AbstractModel):
    def __init__(self, accuracy, mu, sigma, name="", **kwargs):
        super(ModelOracle, self).__init__(name=name, **kwargs)
        self.accuracy = accuracy
        self.mu = mu
        self.sigma = sigma

class ModelTheory(AbstractModel):
    def __init__(self, name="", **kwargs):
        super(ModelTheory, self).__init__(name=name, **kwargs)
        self.accuracies = [True]
        self.inference_times = []
        
        if "ewm" in kwargs and kwargs["ewm"]:
            self._add_infertime_func = self.addInferenceTimeEWM
        else:
            self._add_infertime_func = self.addInferenceTimeEternal
        
        self.first_run = True
        
    def makeObservation(self, new_time, prediction_accuracy=0.5, **kwargs):
        self.addInferenceTime(new_time, **kwargs)
        self.updateAccuracy(prediction_accuracy)
    def updateAccuracy(self, new_accuracy):
        pass
        
    def getLatestTime(self):
        try:
            return self.inference_times[-1]
        except IndexError:
            return self.mu

    def addInferenceTime(self, infer_time):
        self.inference_times.append(infer_time)
        self._add_infertime_func(infer_time)
        if self.first_run:
            self.inference_times = []
            self.first_run = False
        
    def addInferenceTimeEternal(self, infer_time):
        self.mu = np.mean(self.inference_times)
        self.sigma = np.std(self.inference_times)
        if self.sigma == 0:
            self.sigma = self.mu
        
    def addInferenceTimeEWM(self, infer_time, alpha=0.5):
        new_sigma2 = (1-alpha)*(self.sigma**2 + alpha*((infer_time - self.mu)**2))
        new_mu = alpha * infer_time + (1-alpha) * self.mu
        self.sigma = math.sqrt(new_sigma2)
        self.mu = new_mu
    
    def getProbabilityOfSuccess(self, time_target):
        # Use a CDF so it is calculating the possibility that a measure is below
        return scipy.stats.norm(self.mu, self.sigma).cdf(time_target)
    
    def getDifferentialProbability(self, t_l, t_u):
        p_l = self.getProbabilityOfSuccess(t_l)
        p_u = self.getProbabilityOfSuccess(t_u)
        return p_l - (p_l - p_u) + 0.01
    
class ServingModel(ModelTheory):
    
    def __init__(self, model_file, input_layer, output_layer, input_height, input_width=None, accuracy=0.5, window_size=100, **kwargs):
        
        # Add in EWM keyword to ensure that we have an expotential weighted median, so we ignore older times more
        if "ewm" in kwargs:
            ewm = kwargs["ewm"]
        else:
            ewm = True

        self.sess = None
        self.input_operation = None
        self.output_tensor = None

        super(ServingModel, self).__init__(ewm=ewm, **kwargs)
        self.model_file = model_file
        self.model_name = os.path.basename(self.model_file)
        #self.sess, self.input_operation, self.output_operation = ServingModel.getGraphData(model_file, input_layer, output_layer)
        self.input_height = input_height
        self.input_width = input_width if input_width is not None else input_height
        
        self.window_size = window_size
        self.accuracy = accuracy
        
    def __str__(self):
        return "[%s, %.2f, %.2f  (%.2f%%) %s]" % (self.model_name, self.mu, self.sigma, 100.0*self.accuracy, self.uuid)
    
    def getInputHeight(self):
        return self.input_height
    def getInputWidth(self):
        return self.input_width
    def getInputOperation(self):
        return self.input_operation
    def getOutputTensor(self):
        return self.output_tensor
    def getModelName(self):
        return self.model_name
    def getAccuracy(self):
        return self.accuracy
    
    def runInference(self, input_t):
        return None

class TFServingModel(ServingModel):
    tf = __import__('tensorflow')
    def __init__(self, model_file, input_layer, output_layer, input_height, input_width=None, accuracy=0.5, window_size=100, **kwargs):
        
        if "num_extra_layers" in kwargs:
            num_extra_layers = kwargs["num_extra_layers"]
        else:
            num_extra_layers = 0
        
        if "do_load" in kwargs:
            do_load = kwargs["do_load"]
        else:
            do_load = True
        self.loaded = False
        
        
        if "cold_sigma_limit" in kwargs:
            self.cold_sigma_limit = kwargs["kwargs"]
        else:
            self.cold_sigma_limit = 10
        self.was_cold = True

        super(TFServingModel, self).__init__(model_file, input_layer, output_layer, input_height, input_width, accuracy, window_size, **kwargs)
        self.model_file = model_file
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.num_extra_layers = num_extra_layers
        if do_load:
            self.doLoad()
            if ("do_warmup" not in kwargs) or kwargs["do_warmup"]:
                self.warmupInference()
    
    def doLoad(self):
        start_time = time.time()
        self.loaded = True
        self.sess, self.input_operation, self.output_tensor = self.getGraphData(self.model_file, self.input_layer, self.output_layer, self.num_extra_layers)
        return (time.time() - start_time)

    def doUnload(self):
        self.loaded = False
        self.sess = None
        self.input_operation = None
        self.output_tensor = None
    
    def getSession(self):
        return self.sess
    
    #Override method
    def runInference(self, input_t):
        #logging.info("TF Inference")
        if not self.loaded:
            model_load_time = self.doLoad()
        else:
            model_load_time = 0.0

        start_time = time.time()
        result = self.sess.run(self.output_tensor,{self.input_operation.outputs[0]: input_t})
        end_time = time.time()
        inference_time = (end_time - start_time)

        inference_time_in_millis = 1000.0*(inference_time)
        self.was_cold = abs(self.mu - inference_time_in_millis) > (self.cold_sigma_limit * self.sigma)

        self.addInferenceTime( inference_time_in_millis)
        return result, (model_load_time, inference_time)
    
    def warmupInference(self, num_warmups=10):
        for i in range(num_warmups):
            input_t =  np.random.rand(1, self.input_height, self.input_width, 3)
            self.runInference(input_t)
        #result = self.sess.run(self.output_tensor,{self.input_operation.outputs[0]: input_t})
    
    @staticmethod
    def getGraphData(model_file, input_layer, output_layer):
        
        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        
        graph = TFServingModel.tf.Graph()
        graph_def = TFServingModel.tf.GraphDef()

        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            TFServingModel.tf.import_graph_def(graph_def)
            
            sess = TFServingModel.tf.Session(graph=graph)
            input_operation = graph.get_operation_by_name(input_name);
            output_operation = graph.get_operation_by_name(output_name);
            
            prev_operation = output_operation
            last_tensor = prev_operation.outputs[0]
            
            extra_layer_size = 10
            
        
        return (sess, input_operation, last_tensor)

class TestingModel(TFServingModel):
    def __init__(self, *args, **kwargs):
        super(TestingModel, self).__init__(*args, **kwargs)
        
        if "true_accuracy" in kwargs:
            self.true_accuracy = kwargs["true_accuracy"]
        else:
            self.true_accuracy = 0.5
        
        if "slowdown_factor" in kwargs:
            self.slowdown_factor = kwargs["slowdown_factor"]
        else:
            self.slowdown_factor = 0.
        
    def isRight(self):
        return np.random.random() < self.true_accuracy
        
    def updateAccuracy(self, *args, **kwargs):
        self.accuracies.append(self.isRight())
        self.accuracy = self.accuracies.count(True) / float(len(self.accuracies))
    
    #Override method
    def runInference(self, input_t):
        #logging.info("TF Inference")
        start_time = time.time()
        result = self.sess.run(self.output_tensor,{self.input_operation.outputs[0]: input_t})
        run_time = time.time() - start_time
        
        # Sleep to add in latency
        time.sleep(self.slowdown_factor * run_time)
        
        self.addInferenceTime( 1000.0*(time.time() - start_time))
        return result
    
    def changeAccuracy(self, ratio_of_change):
        amount_of_change = (ratio_of_change - 1) * self.true_accuracy
        if amount_of_change > 0:
            self.model_name += "(+%.2f)" % amount_of_change
        else:
            self.model_name += "(%.2f)" % amount_of_change
        self.true_accuracy += amount_of_change
    
    def changeLatency(self, slowdown_change):
        amount_of_change = (slowdown_change - 1) * self.true_accuracy
        if slowdown_change > 0:
            self.model_name += "(+x%s)" % amount_of_change
        else:
            self.model_name += "(+x%s)" % amount_of_change
        self.slowdown_factor += amount_of_change
    
    
    @classmethod
    def generateRangeAccuracy(cls, num_to_generate, *args, **kwargs):
        models = []
        for i in range(num_to_generate):
            ratio = float(i+1) / num_to_generate
            model = cls(*args, **kwargs)
            model.true_accuracy *= ratio
            model.model_name += (" (%.2f)" % (ratio,))
            
            models.append( model )
        
        return models
    
    @classmethod
    def generateRangeLatency(cls, max_slowdown_factor=2, num_slices=10, *args, **kwargs):
        
        models = []
        for i in range(0, num_slices+1):
            slowdown_factor = (float(i) / num_slices) * max_slowdown_factor
            model = cls(*args, slowdown_factor=slowdown_factor, **kwargs)
            model.model_name += (" (x%s)" % (1+slowdown_factor,))
            models.append( model )
        
        return models
    
    
    
    @classmethod
    def generateRegularSpread(cls, num_to_generate, *args, **kwargs):
        '''
        Generates a range of models where both the accuracy and latency increases
        '''
        models = []
        for i in range(1,num_to_generate+1):
            ratio = float(i) / num_to_generate
            model = cls(slowdown_factor=i, *args, **kwargs)
            model.true_accuracy *= ratio
            model.model_name += (" (%.2f, x%s)" % (ratio,i))
            
            models.append( model )
        
        return models
    
    @classmethod
    def generateReverseSpread(cls, num_to_generate, *args, **kwargs):
        '''
        Generates a range of models where both the accuracy decreases as the latency increases
        '''
        models = []
        for i in range(1,num_to_generate+1):
            ratio = float(num_to_generate-i+1) / num_to_generate
            model = cls(slowdown_factor=i, *args, **kwargs)
            model.true_accuracy *= ratio
            model.model_name += (" (%.2f, x%s)" % (ratio,i))
            
            models.append( model )
        
        return models
    


        
        
