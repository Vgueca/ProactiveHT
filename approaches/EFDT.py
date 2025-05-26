

from approaches.base import ProactiveHTBase, register_approach
import copy

@register_approach  
class EFDT(ProactiveHTBase):
    ID = 21
    NAME = "EFDT"

    def __init__(self, window_size=1000, approach=21, stats_reset_mode=0):
        super().__init__(window_size, approach, stats_reset_mode)
        self.window_size = 1000

    
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