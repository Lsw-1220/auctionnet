"""GAS-infer: paper section 3, Algorithm 1 and QT hyperparameters in Table 6.

The policy reward is per-timestep value * CPA penalty (DT-score), with
undiscounted score RTG. QT uses the same reward and gamma=.99 for Bellman targets.
"""
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import default_collate
from bidding_train_env.baseline.dt_baselines.disk_buffer import DiskReplayBuffer
from bidding_train_env.baseline.dt_baselines.dt_baselines import DecisionTransformer
from bidding_train_env.baseline.dt_baselines.dt_critics import DT_Critic


def score_reward(value, cost, cpa):
    """Paper preference 2, beta=2; calculate CPA in raw monetary units."""
    value, cost = np.asarray(value), np.asarray(cost)
    if np.any(value < 0) or np.any(cost < -1e-5) or np.any(np.asarray(cpa) <= 0):
        raise ValueError('Invalid value/cost/CPA for score reward')
    ratio = cpa * value / np.maximum(cost, 1e-10)
    return value * np.minimum(np.maximum(ratio, 0), 1)**2


class PaperReplayBuffer(DiskReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, K=20, **kwargs)
        self.cumulative = np.cumsum(self.traj_lens)

    def window(self, index, start=None):
        traj = self.trajectories[index]
        length = len(traj['observations'])
        start = random.randrange(length) if start is None else start
        # 20 training transitions plus one bootstrap state. Terminal transitions
        # remain in the loss; a nonterminal last token never loses its successor.
        n = min(20, length-start)
        pad = 20-n
        states = np.zeros((21, 16), dtype=np.float32)
        actions = np.zeros((21, 1), dtype=np.float32)
        mask = np.zeros(21, dtype=np.float32)
        timesteps = np.zeros(21, dtype=np.int64)
        normalized = (traj['observations']-self.state_mean)/self.state_std
        count = min(21-pad, length-start)
        states[pad:pad+count] = normalized[start:start+count]
        actions[pad:pad+count] = traj['actions'][start:start+count]
        mask[pad:pad+count] = 1
        timesteps[pad:pad+count] = np.arange(start,start+count)
        rewards = score_reward(traj['rewards'].reshape(-1),traj['cost_ts'],traj['cpa_constrain']) / self.scale
        rtg = np.r_[np.cumsum(rewards[::-1])[::-1],0.]
        conditions = np.zeros((21,1), dtype=np.float32)
        conditions[pad:pad+count,0] = rtg[start:start+count]
        target_rewards = np.zeros((20,1), dtype=np.float32)
        target_rewards[pad:,0] = rewards[start:start+n]
        dones = np.ones(20, dtype=np.float32)
        dones[pad:] = traj['dones'][start:start+n]
        return tuple(torch.from_numpy(x) for x in (states,actions,conditions,timesteps,mask,target_rewards,dones))

    def sample(self, size):
        indices = np.searchsorted(self.cumulative,np.random.randint(self.cumulative[-1],size=size),side='right')
        return default_collate([self.window(int(i)) for i in indices])


class ScorePolicy(DecisionTransformer):
    def __init__(self, state_mean, state_std, device='cpu', target_return=1.):
        super().__init__(16,1,state_mean,state_std,K=20,device=device,
                         baseline_method='vanilla_dt',target_return=target_return)

    @torch.no_grad()
    def take_actions(self, state, actual_excuted_action, target_return=None, target_ctg=None,
                     pre_reward=None, pre_cost=None, cpa_constrain=None):
        if pre_reward is not None:
            pre_reward = float(score_reward(pre_reward,pre_cost,cpa_constrain))
        return super().take_actions(state,actual_excuted_action,target_return,target_ctg,
                                    pre_reward,pre_cost,cpa_constrain)

    def batch_losses(self, batch):
        states,actions,rtg,times,mask,_,_ = batch
        zeros = torch.zeros_like(rtg[:,:20])
        _,predictions,_,_ = self.forward(states[:,:20],actions[:,:20],zeros,rtg[:,:20],
                                         zeros,rtg[:,:20],times[:,:20],mask[:,:20])
        return (((predictions-actions[:,:20])**2)[mask[:,:20]>0].mean(),)


class PaperQT(DT_Critic):
    def __init__(self, state_mean, state_std, device='cpu'):
        super().__init__(16,1,state_mean,state_std,K=20,device=device,use_rtg=False,
                         baseline_method='dt_reweight_search_Q')
        # Table 6: ReLU, AdamW decay .01 and eps 1e-8. Legacy code used GELU.
        for block in self.transformer:
            block.mlp[1] = nn.ReLU()
        self.weight_decay = .01
        self.optimizer = torch.optim.AdamW(self.parameters(),lr=1e-4,weight_decay=.01,eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lambda s:min((s+1)/10000,1))
        for module in (self.critic1_target,self.critic2_target):
            module.requires_grad_(False)

    def batch_losses(self, batch):
        states,actions,rtg,times,mask,rewards,dones = batch
        zeros = torch.zeros_like(rtg)
        _,s,a,_ = self.forward(states,actions,zeros,zeros,zeros,zeros,times,mask)
        # First 20 slots have actual loss targets; slot 21 is only for bootstrap.
        valid = mask[:,:20] > 0
        value_loss = self.calc_value_loss(s[:,:20],a[:,:20],mask[:,:20])
        q1,q2 = self.calc_q_loss(s[:,:20],a[:,:20],rewards,dones,s[:,1:],mask[:,:20])
        if not valid.any():
            raise ValueError('Empty training window')
        return q1,q2,value_loss

    def update_targets(self):
        self.update_target(self.critic1,self.critic1_target)
        self.update_target(self.critic2,self.critic2_target)


def update_batch(model,batch,micro_batch_size):
    device = next(model.parameters()).device
    total = batch[4][:,:20].sum().item()
    model.optimizer.zero_grad(set_to_none=True)
    metrics = None
    for start in range(0,len(batch[0]),micro_batch_size):
        micro = tuple(t[start:start+micro_batch_size].to(device) for t in batch)
        weight = micro[4][:,:20].sum().item()/total
        losses = model.batch_losses(micro)
        (sum(losses)*weight).backward()
        values = np.array([x.detach().item() for x in losses])*weight
        metrics = values if metrics is None else metrics+values
    # Preserve policy's original .25 clipping. Paper QT does not specify clipping.
    if isinstance(model,ScorePolicy):
        torch.nn.utils.clip_grad_norm_(model.parameters(),.25)
    if not np.isfinite(metrics).all():
        raise FloatingPointError(f'Nonfinite losses: {metrics}')
    model.optimizer.step()
    if isinstance(model,PaperQT):
        model.update_targets()
    model.scheduler.step()
    return metrics


def save_checkpoint(model,path,kind,seed,step):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    model.save_net(str(path))
    np.savez(path/'normalization.npz',mean=model.state_mean,std=model.state_std)
    (path/'model.json').write_text(json.dumps(dict(format_version=1,kind=kind,seed=seed,step=step,
        context=20,reward='per_step_score_beta2',target_return=getattr(model,'target_return',1.)),indent=2),encoding='utf-8')


def load_checkpoint(path,device):
    path=Path(path)
    config=json.loads((path/'model.json').read_text(encoding='utf-8'))
    if config.get('format_version') != 1 or config.get('kind') not in ('policy','critic'):
        raise ValueError('Unsupported GAS checkpoint format or model kind')
    with np.load(path/'normalization.npz') as stats:
        cls=ScorePolicy if config['kind']=='policy' else PaperQT
        kwargs=dict(target_return=config['target_return']) if cls is ScorePolicy else {}
        model=cls(stats['mean'].copy(),stats['std'].copy(),device=device,**kwargs).to(device)
    model.load_net(str(path/'dt.pt'),device=device)
    model.eval()
    return model


def q_vote(values):
    """Rows: independently trained QTs, columns: candidates. Constant row abstains."""
    values=np.asarray(values,dtype=float)
    if values.ndim!=2 or not np.isfinite(values).all():
        raise ValueError('Expected finite QT by candidate matrix')
    low=values.min(axis=1,keepdims=True)
    span=values.max(axis=1,keepdims=True)-low
    votes=np.divide(values-low,span,out=np.zeros_like(values),where=span>0)
    return votes.sum(axis=0)


@torch.no_grad()
def search_action(policy,critics,alpha,rng,action_num=5):
    # Put unmodified proposal first so tied/constant votes preserve the policy.
    factors=np.r_[1.,rng.uniform(.9,1.1,action_num-1)].astype(np.float32)
    proposals=factors*float(np.asarray(alpha).reshape(-1)[0])
    values=[]
    for critic in critics:
        device=next(critic.parameters()).device
        s=policy.eval_states[-20:].to(device=device,dtype=torch.float32)
        mean=torch.as_tensor(critic.state_mean,device=device,dtype=torch.float32)
        std=torch.as_tensor(critic.state_std,device=device,dtype=torch.float32)
        s=((s-mean)/std).unsqueeze(0).expand(action_num,-1,-1)
        a=policy.eval_actions[-20:].to(device).unsqueeze(0).repeat(action_num,1,1)
        a[:,-1,0]=torch.as_tensor(proposals,device=device)
        times=policy.eval_timesteps[:,-20:].to(device).expand(action_num,-1)
        zeros=torch.zeros((action_num,s.shape[1],1),device=device)
        mask=torch.ones_like(times)
        _,_,_,qs=critic.forward(s,a,zeros,zeros,zeros,zeros,times,mask)
        values.append(torch.minimum(qs[0][:,-1,0],qs[1][:,-1,0]).cpu().numpy())
    selected=proposals[int(np.argmax(q_vote(values)))]
    # Commit only the chosen action; candidate evaluation never changes history.
    policy.eval_actions[-1,0]=float(selected)
    return np.array([selected],dtype=np.float32)
