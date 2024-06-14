import numpy as np
import xgboost as xgb
import glob
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import random
from imblearn.over_sampling import SMOTE


"""
SIRSYNXGB
"""
def proposed_by_smote(X, y, random_state=0):
      sm = SMOTE(random_state=random_state)
      X_sm, y_sm = sm.fit_resample(X, y)
      z1 = X_sm[y_sm > 0.5, :]
      z0 = X_sm[y_sm < 0.5, :]
      
      return z1, z0


class SirSynXGB:
    def __init__(self, n_estimator=100, proposed_sampler=None, metrics="f1"):
        self.n_estimator=n_estimator
        self.proposed_sampler = proposed_sampler
        if metrics in ["f1", "macro-precision", "g-mean"]:
            self.metrics = metrics
        else:
            raise ValueError("Incorrect specification of metrics")
        
    def calc_weighted_accuracy(self, dtest, bst):
        y = dtest.get_label()
        y_pred = np.round(bst.predict(dtest))
        score = np.mean(precision_recall_fscore_support(y_pred=y_pred, y_true=y,zero_division=0)[1])
        return score
    
    def score(self, dtest):
        y = dtest.get_label()
        y_pred = np.round(self.bst.predict(dtest))
        if self.metrics == "precision":
            score = np.mean(precision_recall_fscore_support(y_pred=y_pred, y_true=y,zero_division=0)[0])
        elif self.metrics == "g-mean":
            t = precision_recall_fscore_support(y_pred=y_pred, y_true=y, zero_division=0)[1]
            score = np.sqrt(t[0]*t[1])
        else:
            score = (precision_recall_fscore_support(y_pred=y_pred, y_true=y, zero_division=0)[2][1])
        
        return score
    
    def calc_weight(self, bst, dtrain):
        y = dtrain.get_label()
        lamda = self.params['reg_lambda']

        prob = bst.predict(dtrain)
        leaf_indices = bst.predict(dtrain, pred_leaf=True)
        if len(leaf_indices.shape) > 1:
            leaf_indices = leaf_indices[:, -1]
        results = {}
        for p, idx in zip(prob, leaf_indices):
            if idx in results:
                results[int(idx)] += p*(1-p)
            else:
                results[int(idx)] = p*(1-p)
        hessian = pd.Series(leaf_indices).map(results).values
        ib = abs(prob - y) / (hessian + lamda + 1e-5)
        weight = (hessian + lamda) / (abs(prob - y)+1e-5)
        return weight, ib
    

    def sampling_sir(self, X, y, proposed_dist, arg:dict, total_sample_size:int=None, random_state=0, y0_sampling=False):
        
        np.random.seed(random_state)
        n = len(y)

        if "random_state" not in arg:
            arg['random_state'] = random_state
        
        # sampling proposal distribution
        z1, z0 = proposed_dist(X,y, **arg)
        dtrain1 = xgb.DMatrix(z1)
        dtrain0 = xgb.DMatrix(z0)
        n1, n0 = len(z1), len(z0)

        if total_sample_size is None:
            total_sample_size = n1 + n0

        if total_sample_size > n1 + n0:
            raise ValueError(f"total_sample_size={total_sample_size} > generated_sample_size = {n1+n0}")
        # calcurate weight
        prob1 = self.bst.predict(dtrain1)
        weight1 = prob1 / (np.sum(prob1) + 1e-10)

        
        if n1 + n0 > 3000:
            create_n1, create_n0 = int(0.8 * total_sample_size/2), int(0.8 * total_sample_size/2)
        else:
            create_n1, create_n0 = int(total_sample_size/2), int(total_sample_size/2)

        random.seed(random_state)
        idx1 = random.choices(range(len(weight1)), k=create_n1, weights=weight1)
        new_X1 = z1[idx1,:]

        if y0_sampling:
            prob0 = (1- self.bst.predict(dtrain0))
            weight0 = prob0 / (np.sum(prob0) + 1e-10)
            idx0 = random.choices(range(len(weight0)), k=create_n0, weights=weight0)
            new_X0 = z0[idx0,:]
        else:
            idx0 = random.choices(range(n0), k=create_n0)
            new_X0 = z0[idx0,:]

        new_X = np.vstack([new_X0, new_X1])
        new_y = np.vstack([ np.zeros(create_n0) , np.ones(create_n1)]).reshape(-1,)

        return new_X, new_y

    def fit(self, dtrain, dtest, params, ealry_stoping=None):
        self.models=[]
        self.params = params
        X_train = dtrain.get_data().toarray()
        y_train = dtrain.get_label()
        self.test_scores = []


        # training
        bst = None
        cnt = 0
        for i in range(1, self.n_estimator+1):
            if i > 1:
                if self.proposed_sampler is None:
                    self.proposed_sampler = proposed_by_smote
                new_X_train, new_y_train = self.sampling_sir(X_train, y_train,
                                                             proposed_dist=self.proposed_sampler,
                                                             arg={"random_state": i},
                                                             total_sample_size=None,
                                                             random_state=i,
                                                             y0_sampling=False)
                dtrain = xgb.DMatrix(new_X_train, label=new_y_train)

            if bst is None:
                bst = xgb.train(params, dtrain, num_boost_round=1)
            else:
                bst = xgb.train(params, dtrain, num_boost_round=1,     
                            xgb_model=bst)
            self.bst = bst

            current_score = self.score(dtest=dtest)
            if len(self.test_scores) > 0 and max(self.test_scores) < current_score:
                cnt = 0
            else:
                cnt +=1
            self.test_scores.append(current_score)

            if cnt > ealry_stoping:
                self.best_n_estimator = i
                
                break

        ndarray = np.array(self.test_scores)
        max_idx = int(np.where(abs(max(ndarray)-ndarray)<1e-10)[0][0])
        self.stop_round=max_idx


        
    def predict(self, dtest, ealry_stoping=True):
        if ealry_stoping:
            idx = self.stop_round
            return self.bst.predict(dtest, iteration_range=(0, idx+1))
        else:
            return self.bst.predict(dtest)