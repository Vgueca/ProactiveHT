
from approaches.base import ProactiveHTBase, register_approach
import copy

@register_approach  
class HoeffdingAdaptiveTree(ProactiveHTBase):
    ID = 20
    NAME = "HoeffdingAdaptiveTree"
    
    def __init__(self, window_size=1000, approach = 20):
        super().__init__(window_size, approach)
        self.window_size = 500 

    
    def train(self, X, y):

        acumulative_DB = []
        list_models = []
        predictions = []

        for idx, (index, row) in enumerate(X.iterrows()):
            x_dict = row.to_dict()
            label = y[index]
    
            y_hat = self.model.predict_one(x_dict)
            predictions.append((label, y_hat))
            self.model.learn_one(x_dict, label)
    
            if (idx+1) % self.window_size == 0 :
                decision_boundaries = self.extract_rules(self.model)
                acumulative_DB.append(decision_boundaries)
                list_models.append(copy.deepcopy(self.model))
        

        decision_boundaries = self.extract_rules(self.model)
        acumulative_DB.append(decision_boundaries)
        list_models.append(copy.deepcopy(self.model))

        return acumulative_DB, list_models, predictions
    
    def update(self, X, y, idx, decision_boundaries):
        pass