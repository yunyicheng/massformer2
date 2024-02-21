import torch as th
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import copy
import math
import pandas as pd
import tempfile
import matplotlib.pyplot as plt


# from massformer.data_utils import ELEMENT_LIST
from configuration_massformer import MassFormerConfig
from collating_massformer import MassFormerBaseDataset, MassFormerDataCollator
from modelling_massformer import MassFormerModel
from src.massformer.utils import *


def get_config():  
    mf_config = MassFormerConfig()
    print(mf_config.__dict__)
    return mf_config


def get_dataset():
    demo_dataset = MassFormerBaseDataset(proc_dp="data/proc_demo/", primary_dset=["mb_na"], secondary_dset=[], 
                           ce_key="nce", inst_type=["FT"], frag_mode=["HCD"], ion_mode="P", process_spec_old=False,
                           pos_prec_type=['[M+H]+', '[M+H-H2O]+', '[M+H-2H2O]+', '[M+2H]2+', '[M+H-NH3]+', "[M+Na]+"],
                           preproc_ce="normalize", mz_max=1000., convert_ce=False, subsample_size=0, num_entries=-1,
                           spectrum_normalization="l1", res=[1,2,3,4,5,6,7], mz_bin_res=1., ints_thresh=0., transform="log10over3")
    return demo_dataset

def get_model(mf_config):
    mf_model = MassFormerModel(config=mf_config)
    print(f"MassFormerModel: {mf_model.__dict__}")
    return mf_model


def get_loss_func(loss_type, mz_bin_res, agg=None):
    # set up loss function
    if loss_type == "mse":
        loss_func = mse
    elif loss_type == "wmse":
        def w_mse(pred, targ):
            weights = compute_weights(pred, mz_bin_res)
            return th.sum(weights * (pred - targ)**2, dim=1)
        loss_func = w_mse
    elif loss_type == "js":
        def js(pred, targ):
            pred = F.normalize(pred, dim=1, p=1)
            targ = F.normalize(targ, dim=1, p=1)
            z = 0.5 * (pred + targ)
            # relu is to prevent NaN from small negative values
            return th.sqrt(F.relu(0.5 * kl(pred, z) + 0.5 * kl(targ, z)))
        loss_func = js
    elif loss_type == "forw_kl":
        # p=targ, q=pred
        def forw_kl(pred, targ):
            pred = F.normalize(pred, dim=1, p=1)
            targ = F.normalize(targ, dim=1, p=1)
            return kl(targ, pred)
        loss_func = forw_kl
    elif loss_type == "rev_kl":
        # p=pred, q=targ
        def rev_kl(pred, targ):
            pred = F.normalize(pred, dim=1, p=1)
            targ = F.normalize(targ, dim=1, p=1)
            return kl(pred, targ)
        loss_func = rev_kl
    elif loss_type == "normal_nll":
        # TBD change this back
        def myloss(input, target):
            from torch.distributions import Normal
            # normalized_input = F.normalize(input, p = 2, dim = 1)
            # normalized_target = F.normalize(target, p = 2, dim = 1)
            nd = Normal(input, 0.08)
            pdf_term = nd.log_prob(target)
            return -th.sum(pdf_term, dim=1)
        loss_func = myloss
    elif loss_type == "wass":
        def wass(pred, targ):
            # does not care about actual m/z distances (just a constant
            # multipler)
            pred = F.normalize(pred, dim=1, p=1)
            targ = F.normalize(targ, dim=1, p=1)
            pred_cdf = th.cumsum(pred, dim=1)
            targ_cdf = th.cumsum(targ, dim=1)
            return th.sum(th.abs(pred_cdf - targ_cdf), dim=1)
        loss_func = wass
    elif loss_type == "cos":
        def cos(pred, targ):
            pred = F.normalize(pred, dim=1, p=2).unsqueeze(1)
            targ = F.normalize(targ, dim=1, p=2).unsqueeze(2)
            return 1. - th.matmul(pred, targ).squeeze(-1).squeeze(-1)
        loss_func = cos
    elif loss_type == "wcos":
        def w_cos(pred, targ):
            weights = compute_weights(pred, mz_bin_res)
            w_pred = F.normalize(weights * pred, dim=1, p=2).unsqueeze(1)
            w_targ = F.normalize(weights * targ, dim=1, p=2).unsqueeze(2)
            return 1. - th.matmul(w_pred, w_targ).squeeze(-1).squeeze(-1)
        loss_func = w_cos
    else:
        raise NotImplementedError
    if not (agg is None):
        if agg == "mean":
            def loss_func_agg(p, t): return th.mean(loss_func(p, t), dim=0)
        elif agg == "sum":
            def loss_func_agg(p, t): return th.sum(loss_func(p, t), dim=0)
    else:
        loss_func_agg = loss_func
    return loss_func_agg

def run_train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_data in progress_bar:
        optimizer.zero_grad()
        # Process batch_data to move to device and forward pass
        batch_data = {k: v.to(device) if isinstance(v, th.Tensor) else v for k, v in batch_data.items()}
        outputs = model(batch_data)
        # Normalize labels and predictions if they are not already
        normalized_labels = F.normalize(batch_data['spec'].float(), p=2, dim=1)
        normalized_preds = F.normalize(outputs['pred'], p=2, dim=1)
        # Get loss
        loss_func = get_loss_func(loss_type="wcos", mz_bin_res=1)
        loss = loss_func(normalized_preds, normalized_labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        progress_bar.set_postfix({'Training Loss': f'{loss.item():.4f}'})
    avg_train_loss = total_train_loss / len(train_loader)
    return avg_train_loss

def run_val(model, val_loader, device):
    model.eval()
    total_val_loss = 0
    progress_bar = tqdm(val_loader, desc='Evaluation')
    with th.no_grad():
        for batch_data in progress_bar:
            # Transfer data to GPU if needed
            batch_data = {k: v.to(device) if isinstance(v, th.Tensor) else v for k, v in batch_data.items()}
            outputs = model(batch_data)
            loss_func = get_loss_func(loss_type="wcos", mz_bin_res=1)
            loss = loss_func(outputs['pred'], batch_data['spec'].float())
            total_val_loss += loss.item()
            progress_bar.set_postfix({'Eval Loss': f'{loss.item():.4f}'})
        # Store the average validation loss
        avg_val_loss = total_val_loss / len(val_loader)
        return avg_val_loss
    
def run_test(model, test_loader, device):
    model.eval()
    total_test_loss = 0
    progress_bar = tqdm(test_loader, desc='Evaluation')
    with th.no_grad():
        for batch_data in progress_bar:
            # Transfer data to GPU if needed
            batch_data = {k: v.to(device) if isinstance(v, th.Tensor) else v for k, v in batch_data.items()}
            outputs = model(batch_data)
            loss_func = get_loss_func(loss_type="wcos", mz_bin_res=1)
            loss = loss_func(outputs['pred'], batch_data['spec'].float())
            total_test_loss += loss.item()
            progress_bar.set_postfix({'Eval Loss': f'{loss.item():.4f}'})
        # Store the average validation loss
        avg_test_loss = total_test_loss / len(test_loader)
        return avg_test_loss


def train_and_eval(model, train_loader, val_loader, optimizer, device, num_epochs=10, patience=20):
    # Training and validation loop
    # Initialize lists to keep track of losses
    train_losses = []
    val_losses = []

    # Early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # Training phase
        avg_losss = run_train_epoch(model=model, train_loader=train_loader, optimizer=optimizer, device=device)
        train_losses.append(avg_losss)
        val_loader = list(val_loader)

        # Validation
        avg_val_loss = run_val(model=model, val_loader=val_loader, device=device)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {avg_val_loss:.4f}')

        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save model, if desired
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

    # When training is done, plot the loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses Over Epochs')
    plt.show()

def main():
    # Initialize the dataset and dataloaders
    ds = get_dataset()

    # Get dataloaders using the method provided by the BaseDataset class
    run_d = {
    "val_frac": 0.0,
    "test_frac": 0.2,
    "sec_frac": 1.00,
    "split_key": "scaffold",
    "split_seed": 42,
    "batch_size": 2,
    "grad_acc_interval": 1,
    "num_workers": 0,
    "optimizer": "adam",
    "pin_memory": True,
    "device": "cuda" if th.cuda.is_available() else "cpu",
    }

    dl_dict, split_id_dict = ds.get_dataloaders(run_d)

    train_loader = dl_dict['train']
    val_loader = dl_dict['primary']['val']
    # Define the model, land optimizer
    mf_config = MassFormerConfig()
    mf_model = MassFormerModel(mf_config)
    if run_d["optimizer"] == "adam":
        optimizer = th.optim.Adam
    elif run_d["optimizer"] == "adamw":
        optimizer = th.optim.AdamW
    else:
        raise NotImplementedError

    # If using a GPU
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    mf_model.to(device)

    train_and_eval(model=mf_model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, device=device, num_epochs=10, patience=20)

if __name__ == "__main__":
    main()