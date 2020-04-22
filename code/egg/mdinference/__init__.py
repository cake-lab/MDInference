import models
import ModelChooser


# Borrowed from https://github.com/requests/requests/blob/master/requests/__init__.py #
# Set default logging handler to avoid "No handler found" warnings.
import logging
try:  # Python 2.7+
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass
logging.getLogger(__name__).addHandler(NullHandler())
#######################################################################################



class Options(object):
    greedy = False
    single_model = None
    latency = False
    accuracy = False
    
    related_greedy = False
    pure_random = False
    related_random = False
    
    # Options that may not exist anymore
    no_probability = False
    aggressive = False
    safe = False
    
    def __init__(self, **kwargs):
        if "greedy" in kwargs and kwargs["greedy"]:
            self.greedy = True
        
        if "no_probability" in kwargs and kwargs["no_probability"]:
            self.no_probability = True
            
        if "aggressive" in kwargs and kwargs["aggressive"]:
            self.aggressive = True
            
        if "safe" in kwargs and kwargs["safe"]:
            self.safe = True
        
        if "single_model" in kwargs:
            self.single_model = kwargs["single_model"]
            if "latency" == kwargs["single_model"]:
                self.latency = True
            elif "accuracy" == kwargs["single_model"]:
                self.accuracy = True
        
        if "related_greedy" in kwargs and kwargs["related_greedy"]:
            self.related_greedy = True
        
        if "pure_random" in kwargs and kwargs["pure_random"]:
            self.pure_random = True
        
        if "related_random" in kwargs and kwargs["related_random"]:
            self.related_random = True
            
    def __str__(self):
        return "Base Options"
    