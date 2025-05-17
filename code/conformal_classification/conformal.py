import numpy as np
from scipy.special import softmax
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
import pandas as pd
import time
from tqdm import tqdm
from conformal_learning.code.conformal_classification.utils import validate, get_logits_targets, sort_sum
import pdb

# Conformalize a model with a calibration set.
# Save it to a file in .cache/modelname
# The only difference is that the forward method of ConformalModel also outputs a set.
class ConformalModel(nn.Module):
    def __init__(self, model, calib_loader, alpha, kreg=None, lamda=None, randomized=True, allow_zero_sets=False, pct_paramtune = 0.3, batch_size=32, lamda_criterion='size'):
        super(ConformalModel, self).__init__()
        self.model = model 
        self.alpha = alpha
        self.T = torch.Tensor([1.3]) #initialize (1.3 is usually a good value)
        self.T, calib_logits = platt(self, calib_loader)
        self.randomized=randomized
        self.allow_zero_sets=allow_zero_sets
        self.num_classes = len(calib_loader.dataset.dataset.classes)

        if kreg == None or lamda == None:
            kreg, lamda, calib_logits = pick_parameters(model, calib_logits, alpha, kreg, lamda, randomized, allow_zero_sets, pct_paramtune, batch_size, lamda_criterion)

        self.penalties = np.zeros((1, self.num_classes))
        self.penalties[:, kreg:] += lamda 

        calib_loader = tdata.DataLoader(calib_logits, batch_size = batch_size, shuffle=False, pin_memory=True)

        self.Qhat = conformal_calibration_logits(self, calib_loader)

    def forward(self, *args, randomized=None, allow_zero_sets=None, **kwargs):
        if randomized == None:
            randomized = self.randomized
        if allow_zero_sets == None:
            allow_zero_sets = self.allow_zero_sets
        logits = self.model(*args, **kwargs)
        
        with torch.no_grad():
            logits_numpy = logits.detach().cpu().numpy()
            scores = softmax(logits_numpy/self.T.item(), axis=1)

            I, ordered, cumsum = sort_sum(scores)

            S = gcq(scores, self.Qhat, I=I, ordered=ordered, cumsum=cumsum, penalties=self.penalties, randomized=randomized, allow_zero_sets=allow_zero_sets)

        return logits, S

# Computes the conformal calibration
def conformal_calibration(cmodel, calib_loader):
    print("Conformal calibration")
    with torch.no_grad():
        E = np.array([])
        for x, targets in tqdm(calib_loader):
            logits = cmodel.model(x.cuda()).detach().cpu().numpy()
            scores = softmax(logits/cmodel.T.item(), axis=1)

            I, ordered, cumsum = sort_sum(scores)

            E = np.concatenate((E,giq(scores,targets,I=I,ordered=ordered,cumsum=cumsum,penalties=cmodel.penalties,randomized=True, allow_zero_sets=True)))
            
        Qhat = np.quantile(E,1-cmodel.alpha,interpolation='higher')

        return Qhat 

# Temperature scaling
def platt(cmodel, calib_loader, max_iters=10, lr=0.01, epsilon=0.01):
    print("Begin Platt scaling.")
    # Save logits so don't need to double compute them
    logits_dataset = get_logits_targets(cmodel.model, calib_loader)
    logits_loader = torch.utils.data.DataLoader(logits_dataset, batch_size = calib_loader.batch_size, shuffle=False, pin_memory=True)

    T = platt_logits(cmodel, logits_loader, max_iters=max_iters, lr=lr, epsilon=epsilon)

    print(f"Optimal T={T.item()}")
    return T, logits_dataset 

"""


        INTERNAL FUNCTIONS


"""

### Precomputed-logit versions of the above functions.

class ConformalModelLogits(nn.Module):
    def __init__(self, model, calib_loader, alpha, kreg=None, lamda=None, randomized=True, allow_zero_sets=False, naive=False, LAC=False, pct_paramtune = 0.3, batch_size=32, lamda_criterion='size'):
        super(ConformalModelLogits, self).__init__()
        self.model = model 
        self.alpha = alpha
        self.randomized = randomized
        self.LAC = LAC
        self.allow_zero_sets = allow_zero_sets
        self.T = platt_logits(self, calib_loader)

        if (kreg == None or lamda == None) and not naive and not LAC:
            kreg, lamda, calib_logits = pick_parameters(model, calib_loader.dataset, alpha, kreg, lamda, randomized, allow_zero_sets, pct_paramtune, batch_size, lamda_criterion)
            calib_loader = tdata.DataLoader(calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True)

        self.penalties = np.zeros((1, calib_loader.dataset[0][0].shape[0]))
        if not (kreg == None) and not naive and not LAC:
            self.penalties[:, kreg:] += lamda
        self.Qhat = 1-alpha
        if not naive and not LAC:
            self.Qhat = conformal_calibration_logits(self, calib_loader)
        elif not naive and LAC:
            gt_locs_cal = np.array([np.where(np.argsort(x[0]).flip(dims=(0,)) == x[1])[0][0] for x in calib_loader.dataset])
            scores_cal = 1-np.array([np.sort(torch.softmax(calib_loader.dataset[i][0]/self.T.item(), dim=0))[::-1][gt_locs_cal[i]] for i in range(len(calib_loader.dataset))]) 
            self.Qhat = np.quantile( scores_cal , np.ceil((scores_cal.shape[0]+1) * (1-alpha)) / scores_cal.shape[0] )

    def forward(self, logits, randomized=None, allow_zero_sets=None):
        if randomized == None:
            randomized = self.randomized
        if allow_zero_sets == None:
            allow_zero_sets = self.allow_zero_sets
        
        with torch.no_grad():
            logits_numpy = logits.detach().cpu().numpy()
            scores = softmax(logits_numpy/self.T.item(), axis=1)

            if not self.LAC:
                I, ordered, cumsum = sort_sum(scores)

                S = gcq(scores, self.Qhat, I=I, ordered=ordered, cumsum=cumsum, penalties=self.penalties, randomized=randomized, allow_zero_sets=allow_zero_sets)
            else:
                S = [ np.where( (1-scores[i,:]) < self.Qhat )[0] for i in range(scores.shape[0]) ]

        return logits, S

def conformal_calibration_logits(cmodel, calib_loader):
    with torch.no_grad():
        E = np.array([])
        for logits, targets in calib_loader:
            logits = logits.detach().cpu().numpy()

            scores = softmax(logits/cmodel.T.item(), axis=1)

            I, ordered, cumsum = sort_sum(scores)

            E = np.concatenate((E,giq(scores,targets,I=I,ordered=ordered,cumsum=cumsum,penalties=cmodel.penalties,randomized=True,allow_zero_sets=True)))
            
        Qhat = np.quantile(E,1-cmodel.alpha,interpolation='higher')

        return Qhat 

def platt_logits(cmodel, calib_loader, max_iters=10, lr=0.01, epsilon=0.01):
    nll_criterion = nn.CrossEntropyLoss().cuda()

    T = nn.Parameter(torch.Tensor([1.3]).cuda())

    optimizer = optim.SGD([T], lr=lr)
    for iter in range(max_iters):
        T_old = T.item()
        for x, targets in calib_loader:
            optimizer.zero_grad()
            x = x.cuda()
            x.requires_grad = True
            out = x/T
            loss = nll_criterion(out, targets.long().cuda())
            loss.backward()
            optimizer.step()
        if abs(T_old - T.item()) < epsilon:
            break
    return T 

### CORE CONFORMAL INFERENCE FUNCTIONS

# Generalized conditional quantile function.
def gcq(scores, tau, I, ordered, cumsum, penalties, randomized, allow_zero_sets):
    penalties_cumsum = np.cumsum(penalties, axis=1)
    sizes_base = ((cumsum + penalties_cumsum) <= tau).sum(axis=1) + 1  # 1 - 1001
    sizes_base = np.minimum(sizes_base, scores.shape[1]) # 1-1000

    if randomized:
        V = np.zeros(sizes_base.shape)
        for i in range(sizes_base.shape[0]):
            V[i] = 1/ordered[i,sizes_base[i]-1] * \
                    (tau-(cumsum[i,sizes_base[i]-1]-ordered[i,sizes_base[i]-1])-penalties_cumsum[0,sizes_base[i]-1]) # -1 since sizes_base \in {1,...,1000}.

        sizes = sizes_base - (np.random.random(V.shape) >= V).astype(int)
    else:
        sizes = sizes_base

    if tau == 1.0:
        sizes[:] = cumsum.shape[1] # always predict max size if alpha==0. (Avoids numerical error.)

    if not allow_zero_sets:
        sizes[sizes == 0] = 1 # allow the user the option to never have empty sets (will lead to incorrect coverage if 1-alpha < model's top-1 accuracy

    S = list()

    # Construct S from equation (5)
    for i in range(I.shape[0]):
        S = S + [I[i,0:sizes[i]],]

    return S

# Get the 'p-value'
def get_tau(score, target, I, ordered, cumsum, penalty, randomized, allow_zero_sets): # For one example
    idx = np.where(I==target)
    tau_nonrandom = cumsum[idx]

    if not randomized:
        return tau_nonrandom + penalty[0]
    
    U = np.random.random()

    if idx == (0,0):
        if not allow_zero_sets:
            return tau_nonrandom + penalty[0]
        else:
            return U * tau_nonrandom + penalty[0] 
    else:
        return U * ordered[idx] + cumsum[(idx[0],idx[1]-1)] + (penalty[0:(idx[1][0]+1)]).sum()

# Gets the histogram of Taus. 
def giq(scores, targets, I, ordered, cumsum, penalties, randomized, allow_zero_sets):
    """
        Generalized inverse quantile conformity score function.
        E from equation (7) in Romano, Sesia, Candes.  Find the minimum tau in [0, 1] such that the correct label enters.
    """
    E = -np.ones((scores.shape[0],))
    for i in range(scores.shape[0]):
        E[i] = get_tau(scores[i:i+1,:],targets[i].item(),I[i:i+1,:],ordered[i:i+1,:],cumsum[i:i+1,:],penalties[0,:],randomized=randomized, allow_zero_sets=allow_zero_sets)

    return E

### AUTOMATIC PARAMETER TUNING FUNCTIONS
def pick_kreg(paramtune_logits, alpha):
    gt_locs_kstar = np.array([np.where(np.argsort(x[0]).flip(dims=(0,)) == x[1])[0][0] for x in paramtune_logits])
    kstar = np.quantile(gt_locs_kstar, 1-alpha, interpolation='higher') + 1
    return kstar 

def pick_lamda_size(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets):
    # Calculate lamda_star
    best_size = iter(paramtune_loader).__next__()[0][1].shape[0] # number of classes 
    # Use the paramtune data to pick lamda.  Does not violate exchangeability.
    for temp_lam in [0.001, 0.01, 0.1, 0.2, 0.5]: # predefined grid, change if more precision desired.
        conformal_model = ConformalModelLogits(model, paramtune_loader, alpha=alpha, kreg=kreg, lamda=temp_lam, randomized=randomized, allow_zero_sets=allow_zero_sets, naive=False)
        top1_avg, top5_avg, cvg_avg, sz_avg = validate(paramtune_loader, conformal_model, print_bool=False)
        if sz_avg < best_size:
            best_size = sz_avg
            lamda_star = temp_lam
    return lamda_star

def pick_lamda_adaptiveness(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets, strata=[[0,1],[2,3],[4,6],[7,10],[11,100],[101,1000]]):
    # Calculate lamda_star
    lamda_star = 0
    best_violation = 1
    # Use the paramtune data to pick lamda.  Does not violate exchangeability.
    for temp_lam in [0, 1e-5, 1e-4, 8e-4, 9e-4, 1e-3, 1.5e-3, 2e-3]: # predefined grid, change if more precision desired.
        conformal_model = ConformalModelLogits(model, paramtune_loader, alpha=alpha, kreg=kreg, lamda=temp_lam, randomized=randomized, allow_zero_sets=allow_zero_sets, naive=False)
        curr_violation = get_violation(conformal_model, paramtune_loader, strata, alpha)
        if curr_violation < best_violation:
            best_violation = curr_violation 
            lamda_star = temp_lam
    return lamda_star

def pick_parameters(model, calib_logits, alpha, kreg, lamda, randomized, allow_zero_sets, pct_paramtune, batch_size, lamda_criterion):
    num_paramtune = int(np.ceil(pct_paramtune * len(calib_logits)))
    paramtune_logits, calib_logits = tdata.random_split(calib_logits, [num_paramtune, len(calib_logits)-num_paramtune])
    calib_loader = tdata.DataLoader(calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True)
    paramtune_loader = tdata.DataLoader(paramtune_logits, batch_size=batch_size, shuffle=False, pin_memory=True)

    if kreg == None:
        kreg = pick_kreg(paramtune_logits, alpha)
    if lamda == None:
        if lamda_criterion == "size":
            lamda = pick_lamda_size(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets)
        elif lamda_criterion == "adaptiveness":
            lamda = pick_lamda_adaptiveness(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets)
    return kreg, lamda, calib_logits

def get_violation(cmodel, loader_paramtune, strata, alpha):
    df = pd.DataFrame(columns=['size', 'correct'])
    for logit, target in loader_paramtune:
        # compute output
        output, S = cmodel(logit) # This is a 'dummy model' which takes logits, for efficiency.
        # measure accuracy and record loss
        size = np.array([x.size for x in S])
        I, _, _ = sort_sum(logit.numpy()) 
        correct = np.zeros_like(size)
        for j in range(correct.shape[0]):
            correct[j] = int( target[j] in list(S[j]) )
        batch_df = pd.DataFrame({'size': size, 'correct': correct})
        df = df.append(batch_df, ignore_index=True)
    wc_violation = 0
    for stratum in strata:
        temp_df = df[ (df['size'] >= stratum[0]) & (df['size'] <= stratum[1]) ]
        if len(temp_df) == 0:
            continue
        stratum_violation = abs(temp_df.correct.mean()-(1-alpha))
        wc_violation = max(wc_violation, stratum_violation)
    return wc_violation # the violation

import numpy as np
from scipy.special import softmax
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
import pandas as pd
import time
from tqdm import tqdm
from conformal_learning.code.conformal_classification.utils import validate, get_logits_targets, sort_sum
import pdb

# Conformalize a model with a calibration set.
# Save it to a file in .cache/modelname
# The only difference is that the forward method of ConformalModel also outputs a set.
class ConformalModel(nn.Module):
    def __init__(self, model, calib_loader, alpha, kreg=None, lamda=None, randomized=True, allow_zero_sets=False, pct_paramtune = 0.3, batch_size=32, lamda_criterion='size'):
        super(ConformalModel, self).__init__()
        self.model = model 
        self.alpha = alpha
        self.T = torch.Tensor([1.3]) #initialize (1.3 is usually a good value)
        self.T, calib_logits = platt(self, calib_loader)
        self.randomized=randomized
        self.allow_zero_sets=allow_zero_sets
        self.num_classes = len(calib_loader.dataset.dataset.classes)

        if kreg == None or lamda == None:
            kreg, lamda, calib_logits = pick_parameters(model, calib_logits, alpha, kreg, lamda, randomized, allow_zero_sets, pct_paramtune, batch_size, lamda_criterion)

        self.penalties = np.zeros((1, self.num_classes))
        self.penalties[:, kreg:] += lamda 

        calib_loader = tdata.DataLoader(calib_logits, batch_size = batch_size, shuffle=False, pin_memory=True)

        self.Qhat = conformal_calibration_logits(self, calib_loader)

    def forward(self, *args, randomized=None, allow_zero_sets=None, **kwargs):
        if randomized == None:
            randomized = self.randomized
        if allow_zero_sets == None:
            allow_zero_sets = self.allow_zero_sets
        logits = self.model(*args, **kwargs)
        
        with torch.no_grad():
            logits_numpy = logits.detach().cpu().numpy()
            scores = softmax(logits_numpy/self.T.item(), axis=1)

            I, ordered, cumsum = sort_sum(scores)

            S = gcq(scores, self.Qhat, I=I, ordered=ordered, cumsum=cumsum, penalties=self.penalties, randomized=randomized, allow_zero_sets=allow_zero_sets)

        return logits, S

# Computes the conformal calibration
def conformal_calibration(cmodel, calib_loader):
    print("Conformal calibration")
    with torch.no_grad():
        E = np.array([])
        for x, targets in tqdm(calib_loader):
            logits = cmodel.model(x.cuda()).detach().cpu().numpy()
            scores = softmax(logits/cmodel.T.item(), axis=1)

            I, ordered, cumsum = sort_sum(scores)

            E = np.concatenate((E,giq(scores,targets,I=I,ordered=ordered,cumsum=cumsum,penalties=cmodel.penalties,randomized=True, allow_zero_sets=True)))
            
        Qhat = np.quantile(E,1-cmodel.alpha,interpolation='higher')

        return Qhat 

# Temperature scaling
def platt(cmodel, calib_loader, max_iters=10, lr=0.01, epsilon=0.01):
    print("Begin Platt scaling.")
    # Save logits so don't need to double compute them
    logits_dataset = get_logits_targets(cmodel.model, calib_loader)
    logits_loader = torch.utils.data.DataLoader(logits_dataset, batch_size = calib_loader.batch_size, shuffle=False, pin_memory=True)

    T = platt_logits(cmodel, logits_loader, max_iters=max_iters, lr=lr, epsilon=epsilon)

    print(f"Optimal T={T.item()}")
    return T, logits_dataset 

"""


        INTERNAL FUNCTIONS


"""

### Precomputed-logit versions of the above functions.

class ConformalModelLogits(nn.Module):
    def __init__(self, model, calib_loader, alpha, kreg=None, lamda=None, randomized=True, allow_zero_sets=False, naive=False, LAC=False, pct_paramtune = 0.3, batch_size=32, lamda_criterion='size'):
        super(ConformalModelLogits, self).__init__()
        self.model = model 
        self.alpha = alpha
        self.randomized = randomized
        self.LAC = LAC
        self.allow_zero_sets = allow_zero_sets
        self.T = platt_logits(self, calib_loader)

        if (kreg == None or lamda == None) and not naive and not LAC:
            kreg, lamda, calib_logits = pick_parameters(model, calib_loader.dataset, alpha, kreg, lamda, randomized, allow_zero_sets, pct_paramtune, batch_size, lamda_criterion)
            calib_loader = tdata.DataLoader(calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True)

        self.penalties = np.zeros((1, calib_loader.dataset[0][0].shape[0]))
        if not (kreg == None) and not naive and not LAC:
            self.penalties[:, kreg:] += lamda
        self.Qhat = 1-alpha
        if not naive and not LAC:
            self.Qhat = conformal_calibration_logits(self, calib_loader)
        elif not naive and LAC:
            gt_locs_cal = np.array([np.where(np.argsort(x[0]).flip(dims=(0,)) == x[1])[0][0] for x in calib_loader.dataset])
            scores_cal = 1-np.array([np.sort(torch.softmax(calib_loader.dataset[i][0]/self.T.item(), dim=0))[::-1][gt_locs_cal[i]] for i in range(len(calib_loader.dataset))]) 
            self.Qhat = np.quantile( scores_cal , np.ceil((scores_cal.shape[0]+1) * (1-alpha)) / scores_cal.shape[0] )

    def forward(self, logits, randomized=None, allow_zero_sets=None):
        if randomized == None:
            randomized = self.randomized
        if allow_zero_sets == None:
            allow_zero_sets = self.allow_zero_sets
        
        with torch.no_grad():
            logits_numpy = logits.detach().cpu().numpy()
            scores = softmax(logits_numpy/self.T.item(), axis=1)

            if not self.LAC:
                I, ordered, cumsum = sort_sum(scores)

                S = gcq(scores, self.Qhat, I=I, ordered=ordered, cumsum=cumsum, penalties=self.penalties, randomized=randomized, allow_zero_sets=allow_zero_sets)
            else:
                S = [ np.where( (1-scores[i,:]) < self.Qhat )[0] for i in range(scores.shape[0]) ]

        return logits, S

def conformal_calibration_logits(cmodel, calib_loader):
    with torch.no_grad():
        E = np.array([])
        for logits, targets in calib_loader:
            logits = logits.detach().cpu().numpy()

            scores = softmax(logits/cmodel.T.item(), axis=1)

            I, ordered, cumsum = sort_sum(scores)

            E = np.concatenate((E,giq(scores,targets,I=I,ordered=ordered,cumsum=cumsum,penalties=cmodel.penalties,randomized=True,allow_zero_sets=True)))
            
        Qhat = np.quantile(E,1-cmodel.alpha,interpolation='higher')

        return Qhat 

def platt_logits(cmodel, calib_loader, max_iters=10, lr=0.01, epsilon=0.01):
    nll_criterion = nn.CrossEntropyLoss().cuda()

    T = nn.Parameter(torch.Tensor([1.3]).cuda())

    optimizer = optim.SGD([T], lr=lr)
    for iter in range(max_iters):
        T_old = T.item()
        for x, targets in calib_loader:
            optimizer.zero_grad()
            x = x.cuda()
            x.requires_grad = True
            out = x/T
            loss = nll_criterion(out, targets.long().cuda())
            loss.backward()
            optimizer.step()
        if abs(T_old - T.item()) < epsilon:
            break
    return T 

### CORE CONFORMAL INFERENCE FUNCTIONS

# Generalized conditional quantile function.
def gcq(scores, tau, I, ordered, cumsum, penalties, randomized, allow_zero_sets):
    penalties_cumsum = np.cumsum(penalties, axis=1)
    sizes_base = ((cumsum + penalties_cumsum) <= tau).sum(axis=1) + 1  # 1 - 1001
    sizes_base = np.minimum(sizes_base, scores.shape[1]) # 1-1000

    if randomized:
        V = np.zeros(sizes_base.shape)
        for i in range(sizes_base.shape[0]):
            V[i] = 1/ordered[i,sizes_base[i]-1] * \
                    (tau-(cumsum[i,sizes_base[i]-1]-ordered[i,sizes_base[i]-1])-penalties_cumsum[0,sizes_base[i]-1]) # -1 since sizes_base \in {1,...,1000}.

        sizes = sizes_base - (np.random.random(V.shape) >= V).astype(int)
    else:
        sizes = sizes_base

    if tau == 1.0:
        sizes[:] = cumsum.shape[1] # always predict max size if alpha==0. (Avoids numerical error.)

    if not allow_zero_sets:
        sizes[sizes == 0] = 1 # allow the user the option to never have empty sets (will lead to incorrect coverage if 1-alpha < model's top-1 accuracy

    S = list()

    # Construct S from equation (5)
    for i in range(I.shape[0]):
        S = S + [I[i,0:sizes[i]],]

    return S

# Get the 'p-value'
def get_tau(score, target, I, ordered, cumsum, penalty, randomized, allow_zero_sets): # For one example
    idx = np.where(I==target)
    tau_nonrandom = cumsum[idx]

    if not randomized:
        return tau_nonrandom + penalty[0]
    
    U = np.random.random()

    if idx == (0,0):
        if not allow_zero_sets:
            return tau_nonrandom + penalty[0]
        else:
            return U * tau_nonrandom + penalty[0] 
    else:
        return U * ordered[idx] + cumsum[(idx[0],idx[1]-1)] + (penalty[0:(idx[1][0]+1)]).sum()

# Gets the histogram of Taus. 
def giq(scores, targets, I, ordered, cumsum, penalties, randomized, allow_zero_sets):
    """
        Generalized inverse quantile conformity score function.
        E from equation (7) in Romano, Sesia, Candes.  Find the minimum tau in [0, 1] such that the correct label enters.
    """
    E = -np.ones((scores.shape[0],))
    for i in range(scores.shape[0]):
        E[i] = get_tau(scores[i:i+1,:],targets[i].item(),I[i:i+1,:],ordered[i:i+1,:],cumsum[i:i+1,:],penalties[0,:],randomized=randomized, allow_zero_sets=allow_zero_sets)

    return E

### AUTOMATIC PARAMETER TUNING FUNCTIONS
def pick_kreg(paramtune_logits, alpha):
    gt_locs_kstar = np.array([np.where(np.argsort(x[0]).flip(dims=(0,)) == x[1])[0][0] for x in paramtune_logits])
    kstar = np.quantile(gt_locs_kstar, 1-alpha, interpolation='higher') + 1
    return kstar 

def pick_lamda_size(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets):
    # Calculate lamda_star
    best_size = iter(paramtune_loader).__next__()[0][1].shape[0] # number of classes 
    # Use the paramtune data to pick lamda.  Does not violate exchangeability.
    for temp_lam in [0.001, 0.01, 0.1, 0.2, 0.5]: # predefined grid, change if more precision desired.
        conformal_model = ConformalModelLogits(model, paramtune_loader, alpha=alpha, kreg=kreg, lamda=temp_lam, randomized=randomized, allow_zero_sets=allow_zero_sets, naive=False)
        top1_avg, top5_avg, cvg_avg, sz_avg = validate(paramtune_loader, conformal_model, print_bool=False)
        if sz_avg < best_size:
            best_size = sz_avg
            lamda_star = temp_lam
    return lamda_star

def pick_lamda_adaptiveness(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets, strata=[[0,1],[2,3],[4,6],[7,10],[11,100],[101,1000]]):
    # Calculate lamda_star
    lamda_star = 0
    best_violation = 1
    # Use the paramtune data to pick lamda.  Does not violate exchangeability.
    for temp_lam in [0, 1e-5, 1e-4, 8e-4, 9e-4, 1e-3, 1.5e-3, 2e-3]: # predefined grid, change if more precision desired.
        conformal_model = ConformalModelLogits(model, paramtune_loader, alpha=alpha, kreg=kreg, lamda=temp_lam, randomized=randomized, allow_zero_sets=allow_zero_sets, naive=False)
        curr_violation = get_violation(conformal_model, paramtune_loader, strata, alpha)
        if curr_violation < best_violation:
            best_violation = curr_violation 
            lamda_star = temp_lam
    return lamda_star

def pick_parameters(model, calib_logits, alpha, kreg, lamda, randomized, allow_zero_sets, pct_paramtune, batch_size, lamda_criterion):
    num_paramtune = int(np.ceil(pct_paramtune * len(calib_logits)))
    paramtune_logits, calib_logits = tdata.random_split(calib_logits, [num_paramtune, len(calib_logits)-num_paramtune])
    calib_loader = tdata.DataLoader(calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True)
    paramtune_loader = tdata.DataLoader(paramtune_logits, batch_size=batch_size, shuffle=False, pin_memory=True)

    if kreg == None:
        kreg = pick_kreg(paramtune_logits, alpha)
    if lamda == None:
        if lamda_criterion == "size":
            lamda = pick_lamda_size(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets)
        elif lamda_criterion == "adaptiveness":
            lamda = pick_lamda_adaptiveness(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets)
    return kreg, lamda, calib_logits

def get_violation(cmodel, loader_paramtune, strata, alpha):
    df = pd.DataFrame(columns=['size', 'correct'])
    for logit, target in loader_paramtune:
        # compute output
        output, S = cmodel(logit) # This is a 'dummy model' which takes logits, for efficiency.
        # measure accuracy and record loss
        size = np.array([x.size for x in S])
        I, _, _ = sort_sum(logit.numpy()) 
        correct = np.zeros_like(size)
        for j in range(correct.shape[0]):
            correct[j] = int( target[j] in list(S[j]) )
        batch_df = pd.DataFrame({'size': size, 'correct': correct})
        df = df.append(batch_df, ignore_index=True)
    wc_violation = 0
    for stratum in strata:
        temp_df = df[ (df['size'] >= stratum[0]) & (df['size'] <= stratum[1]) ]
        if len(temp_df) == 0:
            continue
        stratum_violation = abs(temp_df.correct.mean()-(1-alpha))
        wc_violation = max(wc_violation, stratum_violation)
    return wc_violation # the violation

# 修改温度缩放函数（关键修复）
def platt_logits_multi(cmodel, calib_loader, max_iters=10):
    """多标签温度缩放（修复数据形状问题）"""
    T = nn.Parameter(torch.tensor([1.0]).cuda())
    optimizer = optim.LBFGS([T], lr=0.01)
    
    # 预计算所有logits
    logits_list = []
    targets_list = []
    with torch.no_grad():
        for inputs, targets in calib_loader:
            logits = cmodel.model(inputs.cuda())
            logits_list.append(logits.cpu())
            targets_list.append(targets)
    
    # 创建新的logits数据集
    calib_logits = torch.utils.data.TensorDataset(
        torch.cat(logits_list), 
        torch.cat(targets_list)
    )
    logits_loader = torch.utils.data.DataLoader(calib_logits, batch_size=calib_loader.batch_size)
    
    criterion = nn.BCEWithLogitsLoss()
    
    def closure():
        optimizer.zero_grad()
        loss = 0
        for logits, targets in logits_loader:
            logits = logits.cuda()
            targets = targets.cuda()
            scaled = logits / T
            loss += criterion(scaled, targets)
        loss.backward()
        return loss
    
    for _ in range(max_iters):
        optimizer.step(closure)
    return T.detach(), calib_logits  # 返回logits数据集

from scipy.special import expit as sigmoid 

# 修改校准函数
def conformal_calibration_logits_multi(cmodel, calib_loader):
    """多标签校准（修复形状匹配）"""
    E_matrix = []
    for logits, targets in calib_loader:
        logits = logits.numpy()
        scores = sigmoid(logits/cmodel.T.item())  # 使用预计算的logits
        
        # 验证形状一致性
        assert scores.shape == targets.shape, \
            f"Shape mismatch: scores {scores.shape} vs targets {targets.shape}"
        
        nonconformity = np.where(targets==1, 1-scores, scores)
        E_matrix.append(nonconformity)
    
    E_matrix = np.concatenate(E_matrix)
    Qhat = np.quantile(E_matrix, 1-cmodel.alpha, axis=0) 
    return Qhat

# 修改模型初始化
class ConformalModelMulti(nn.Module):
    def __init__(self, model, calib_loader, alpha, **kwargs):
        super().__init__()
        self.model = model
        self.alpha = alpha
        
        # 温度缩放（返回logits数据集）
        self.T, calib_logits = platt_logits_multi(self, calib_loader)
        
        # 创建仅包含logits的loader
        logits_loader = tdata.DataLoader(
            calib_logits, 
            batch_size=32, 
            shuffle=False,
            collate_fn=lambda x: (x[0][0], x[0][1])  # 解包TensorDataset
        )
        
        # 多标签校准
        self.Qhat = conformal_calibration_logits_multi(self, logits_loader)

    def forward(self, x):
        logits = self.model(x)
        # 修复GPU张量转换问题
        scores = torch.sigmoid(logits/self.T.item()).cpu().detach().numpy()  # 添加.cpu().detach()
        
        # 多标签预测集合生成
        S = []
        for i in range(scores.shape[0]):
            # 确保阈值形状匹配 (8,)
            class_thresholds = 1 - self.Qhat.reshape(-1)  # 展平阈值
            pred_mask = scores[i] > class_thresholds
            S.append(np.where(pred_mask)[0].tolist())
        
        return logits, S
