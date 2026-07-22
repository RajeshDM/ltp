import copy
import os
import time
import warnings
import wandb

import numpy as np
from torch._C import device
import pddlgym
import torch
import torch.nn as nn

# GradScaler moved from torch.cuda.amp (<=2.2) to torch.amp (>=2.4). Import
# defensively so the same code runs on both the 2.6 and 2.2 environments
# (autocast lives at torch.amp.autocast in both, so it needs no shim).
try:
    from torch.amp import GradScaler as _GradScaler
except (ImportError, AttributeError):
    from torch.cuda.amp import GradScaler as _GradScaler

import ploi.constants as constants
import matplotlib.pyplot as plt
from icecream import ic
from torchviz import make_dot
from ploi.profile_manager import (
    ProfilerManager,
    ConditionalRecordFunction
)

import os
import sys

from .planning import PlanningFailure, PlanningTimeout, validate_strips_plan


def compute_val_loss(state_val, batch_data):
    target_state_val = batch_data['goal_dist'].x/100
    target_state_val = target_state_val.to(state_val[0].dtype)
    loss = torch.mean(torch.abs(torch.sub(target_state_val, state_val)))
    return loss

def save_model_graphnetwork(model, save_folder, epoch, optimizer,train_env_name,
                             seed, message_string, best_seen_running_validation_loss,
                             running_loss, best_seen_model_weights, best_validation_loss_epoch,
                             time_taken_for_save_iter):
    save_path = os.path.join(save_folder, str(train_env_name) + "_seed" + str(seed) + "_model" + str(
        epoch) + "_" + message_string + ".pt")
    state_save = {'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    'epochs':epoch}
    torch.save(state_save,save_path)
    if running_loss['val'] < best_seen_running_validation_loss:
        best_seen_running_validation_loss = running_loss['val']
        best_seen_model_weights = model.state_dict()
        #print("Found new best model with validation loss {} at epoch {}".format(
        #    best_seen_running_validation_loss, epoch), flush=True)
        best_validation_loss_epoch = epoch
        #print ()
        print("Saved model checkpoint {}, Time : {} , (New Best)".format(save_path,time.time()-time_taken_for_save_iter))
    else :
        print("Saved model checkpoint {}, Time : {}".format(save_path,time.time()-time_taken_for_save_iter))

    return best_seen_running_validation_loss,best_validation_loss_epoch, best_seen_model_weights

def train_model_graphnetwork_ltp_batch_val(model, datasets,
                                 #dataloaders,
                                   criterion, optimizer, use_gpu, print_iter=10,
                save_iter=100, save_folder='/tmp',starting_epoch=0, final_epoch=1000, global_criterion=None,
                return_last_model_weights=True,dagger_train=False,train_env_name=None,seed=None,
                message_string='',
                log_wandb=False,
                ablation="val",
                chpkt_manager=None,
                enable_profiling=False,
                use_amp=False,
                spot_checkpoint_path=None,
                patience=0,
                domain_names=None):

    since = time.time()
    min_save_epoch = 0
    print_iter = 10
    save_iter = print_iter
    if use_gpu:
        model = model.cuda()
        device = "cuda:0"
        if criterion is not None:
            criterion = criterion.cuda()
    else:
        device = "cpu"

    device_type = 'cuda' if use_gpu else 'cpu'
    scaler = _GradScaler(enabled=use_amp and use_gpu)

    epochs = []
    train_loss_values = []
    val_loss_values = []
    time_taken_for_save_iter = time.time()
    _best_val_for_patience = float('inf')
    _patience_counter = 0
    for epoch in range(starting_epoch,final_epoch+1):
        if epoch % print_iter == 0:
            print(f"Epoch {epoch}/{final_epoch}", end=" ", flush=True)
        # Each epoch has a training and validation phase
        running_num_samples = 0
        if epoch % print_iter == 0 :
            phases = ['train','val']
        else:
            phases = ['train']

        running_loss = {'train':0.0,'val':0.0}

        for phase in phases:
            if phase == 'train':
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()

            for i,batch_data in enumerate(datasets[phase]):
                loss = 0.
                optimizer.zero_grad()
                batch_data = batch_data.to(device)
                with torch.amp.autocast(device_type, enabled=use_amp):
                    state_val =  model(batch_data).squeeze(1)
                    loss += compute_val_loss(state_val, batch_data)

                if phase == 'train':
                    backward_time = time.time()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                # statistics
                running_loss[phase] += loss.item()
                running_num_samples += 1
            if log_wandb:
                wandb.log({f"loss_{phase}": running_loss[phase]})

        if epoch % print_iter == 0:
            print(f"loss: {running_loss} | time({save_iter}): {time.time() - time_taken_for_save_iter:.2f}s", flush=True)
            epochs.append(epoch)
            train_loss_values.append(running_loss['train'])
            val_loss_values.append(running_loss['val'])

            if patience > 0:
                if running_loss['val'] < _best_val_for_patience:
                    _best_val_for_patience = running_loss['val']
                    _patience_counter = 0
                else:
                    _patience_counter += 1
                    if _patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch} "
                              f"(no val improvement for {patience} checks)", flush=True)
                        break

        if epoch % save_iter == 0 and epoch >= min_save_epoch:
            chpkt_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_env_name=train_env_name,
                seed=42,
                losses={'train': running_loss["train"], 'val': running_loss["val"]},
            )
            time_taken_for_save_iter = time.time()

        if spot_checkpoint_path is not None:
            from ploi.run_modes import save_spot_checkpoint
            save_spot_checkpoint(spot_checkpoint_path, model, optimizer, epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60), flush=True)

def train_model_graphnetwork_ltp_batch_val_profiling(model, datasets,
                                 criterion, optimizer, use_gpu, print_iter=10,
                                 save_iter=100, save_folder='/tmp', starting_epoch=0, final_epoch=1000,
                                 global_criterion=None, return_last_model_weights=True, dagger_train=False,
                                 train_env_name=None, seed=None, message_string='',
                                 log_wandb=False,
                                 ablation="val",
                                 chpkt_manager=None,
                                 enable_profiling=True,
                                 use_amp=False,
                                 spot_checkpoint_path=None,
                                 patience=0,
                                 domain_names=None,
                                 profile_log_dir='./profile_logs'):

    since = time.time()
    min_save_epoch = 0
    save_iter = print_iter
    
    # Initialize the profiler manager if profiling is enabled
    profiler = ProfilerManager(log_dir=profile_log_dir) if enable_profiling else None
    
    if use_gpu:
        model = model.cuda()
        device = "cuda:0"
        if criterion is not None:
            criterion = criterion.cuda()
    else:
        device = "cpu"

    epochs = []
    train_loss_values = []
    val_loss_values = []
    time_taken_for_save_iter = time.time()
    
    for epoch in range(starting_epoch, final_epoch+1):
        epoch_start_time = time.time()
        
        # Determine if this epoch should have detailed output and profiling
        should_detail = epoch % print_iter == 0
        
        # Start profiling if enabled and this is a detail epoch
        if enable_profiling and should_detail:
            profiler.start_profiling_session(epoch)
        
        if should_detail:
            print('Epoch {}/{}'.format(epoch, final_epoch), flush=True)
            print('-' * 10, flush=True)
        
        # Each epoch has a training and validation phase
        running_num_samples = 0
        phases = ['train', 'val'] if should_detail else ['train']
        running_loss = {'train': 0.0, 'val': 0.0}

        for phase in phases:
            phase_start_time = time.time()
            
            # Set model mode based on phase
            model.train() if phase == 'train' else model.eval()

            for i, batch_data in enumerate(datasets[phase]):
                # Process batch with optional profiling
                with ConditionalRecordFunction(f"{phase}_batch", enable=enable_profiling and should_detail):
                    batch_start_time = time.time()
                    
                    loss = 0.
                    optimizer.zero_grad()
                    batch_data = batch_data.to(device)
                    
                    # Forward pass
                    with ConditionalRecordFunction("forward", enable=enable_profiling and should_detail):
                        state_val = model(batch_data).squeeze(1)
                    
                    # Loss calculation
                    with ConditionalRecordFunction("loss_calculation", enable=enable_profiling and should_detail):
                        target_state_val = batch_data['goal_dist'].x/100
                        target_state_val = target_state_val.to(state_val[0].dtype)
                        loss += torch.mean(torch.abs(torch.sub(target_state_val, state_val)))

                    # Backward pass for training
                    if phase == 'train':
                        with ConditionalRecordFunction("backward", enable=enable_profiling and should_detail):
                            backward_time = time.time()
                            loss.backward()
                            optimizer.step()
                            backward_duration = time.time() - backward_time
                            
                            if should_detail and i == 0:
                                print(f"Backward pass time: {backward_duration:.4f}s")

                    # Statistics
                    running_loss[phase] += loss.item()
                    running_num_samples += 1
                    
                    # Print batch timing info for first few batches in detail epochs
                    if should_detail and i < 3:
                        batch_duration = time.time() - batch_start_time
                        print(f"  {phase} batch {i} time: {batch_duration:.4f}s")
                
                # Step the profiler after each batch if active
                if enable_profiling and should_detail:
                    profiler.step()
            
            # Print phase timing at detail epochs
            if should_detail:
                phase_duration = time.time() - phase_start_time
                print(f"  {phase} phase completed in {phase_duration:.2f}s")
            
            # Log to wandb if enabled
            if log_wandb:
                wandb.log({f"loss_{phase}": running_loss[phase]})

        # Print epoch summary at detail epochs
        if should_detail:
            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch {epoch} completed in {epoch_duration:.2f}s")
            print(f"Running loss: {running_loss}", flush=True)
            
            epochs.append(epoch)
            train_loss_values.append(running_loss['train'])
            val_loss_values.append(running_loss['val'])
            
            # End profiling session if active
            if enable_profiling:
                profiler.end_profiling_session()
    
        # Save checkpoint at save_iter intervals
        if epoch % save_iter == 0 and epoch >= min_save_epoch:
            checkpoint_start_time = time.time()
            chpkt_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_env_name=train_env_name,
                seed=42,
                losses={'train': running_loss["train"], 'val': running_loss["val"]},
            )
            checkpoint_duration = time.time() - checkpoint_start_time
            
            if should_detail:
                print(f"Checkpoint saved in {checkpoint_duration:.2f}s")
                print(f"Time taken for {save_iter} epochs: {time.time() - time_taken_for_save_iter:.2f}s")
            
            time_taken_for_save_iter = time.time()

    # Training complete
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60), flush=True)
    
    # Ensure profiler is stopped if active
    if enable_profiling and profiler and profiler.is_profiling:
        profiler.end_profiling_session()
        
    return model

def train_model_graphnetwork_ltp_batch(model, datasets,
                                 #dataloaders,
                                   criterion, optimizer, use_gpu, print_iter=10,
                save_iter=100, save_folder='/tmp',starting_epoch=0, final_epoch=1000, global_criterion=None,
                return_last_model_weights=True,dagger_train=False,train_env_name=None,seed=None,
                message_string='',
                log_wandb=False,
                ablation="main",
                chpkt_manager=None,
                enable_profiling=False,
                use_amp=False,
                spot_checkpoint_path=None,
                patience=0,
                domain_names=None):

    since = time.time()
    min_save_epoch = 0
    print_iter = 10
    save_iter = print_iter
    if use_gpu:
        model = model.cuda()
        device = "cuda:0"
        if criterion is not None:
            criterion = criterion.cuda()
    else:
        device = "cpu"

    epochs = []
    train_loss_values = []
    val_loss_values = []
    time_taken_for_save_iter = time.time()
    for epoch in range(starting_epoch,final_epoch+1):
        if epoch % print_iter == 0:
            print(f"Epoch {epoch}/{final_epoch}", end=" ", flush=True)
        # Each epoch has a training and validation phase
        running_num_samples = 0
        if epoch % print_iter == 0 :
            phases = ['train','val']
        else:
            phases = ['train']

        running_loss = {'train':0.0,'val':0.0}

        for phase in phases:
            if phase == 'train':
                # Set model to training mode
                model.train()  
            else:
                # Set model to evaluate mode
                model.eval()

            for i,batch_data in enumerate(datasets[phase]):
                optimizer.zero_grad()
                batch_data = batch_data.to(device)
                action_scores, action_object_scores = model(batch_data, beam_search=False)
                tgt_action_scores = batch_data['target_action_scores'].x
                tgt_action_object_scores = batch_data['target_action_object_scores'].x
                tgt_params = batch_data['target_n_parameters'].x
                loss = 0.
                curr_param_counter = 0
                required_action_object_scores = []
                total_number_params = 0

                for idx,n_params in enumerate(tgt_params):
                    #ic (output['action_object_scores'][0:2])
                    #ic (curr_param_counter)
                    n_params = int(n_params)
                    #ic (n_params)
                    for correct_index in range(curr_param_counter,curr_param_counter+n_params):
                        required_action_object_scores.append(correct_index)
                    #loss += criterion(output['action_object_scores'][curr_param_counter:curr_param_counter+n_params], targets['action_object_scores'][curr_param_counter:curr_param_counter+n_params])
                    #TODO ADD an assert here to check if any of the elements in target are all zeroes
                    #ic (output['action_object_scores'][curr_param_counter:curr_param_counter+n_params])
                    #ic (targets['action_object_scores'][curr_param_counter:curr_param_counter+n_params])
                    #curr_param_counter += 2
                    curr_param_counter += model.max_number_action_parameters
                    total_number_params += n_params

                required_action_object_scores = torch.tensor(required_action_object_scores)
                target_indices = tgt_action_scores.argmax(dim=1)
                target_indices_2 = tgt_action_object_scores[required_action_object_scores].argmax(dim=1)
                tgt_action_scores = tgt_action_scores.squeeze(0)

                m = torch.nn.ConstantPad2d((0,tgt_action_object_scores.shape[1]-action_object_scores.shape[1]\
                                            ,0,0),0)
                
                action_object_scores = m(action_object_scores)
                #loss += criterion(output['action_object_scores'][required_action_object_scores],targets['action_object_scores'][required_action_object_scores])/division_coeff
                loss += criterion(action_scores,target_indices)
                loss += criterion(action_object_scores[required_action_object_scores],target_indices_2)

                if phase == 'train':
                    backward_time = time.time()
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    #ic ("backprop time",time.time()-backward_time)

                # statistics
                running_loss[phase] += loss.item()
                running_num_samples += 1
            if log_wandb:
                wandb.log({f"loss_{phase}": running_loss[phase]})

        if epoch % print_iter == 0:
            #print("running_loss:", running_loss, flush=True)
            print(f"loss: {running_loss} | time({save_iter}): {time.time() - time_taken_for_save_iter:.2f}s", flush=True)
            epochs.append(epoch)
            train_loss_values.append(running_loss['train'])
            val_loss_values.append(running_loss['val'])
    
        if epoch % save_iter == 0 and epoch >= min_save_epoch:
            chpkt_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_env_name=train_env_name,
                seed=42,
                losses={'train': running_loss["train"], 'val': running_loss["val"]},
            )
            #print ("Time taken for {} epochs : {}".format(save_iter, time.time() - time_taken_for_save_iter))
            time_taken_for_save_iter = time.time()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60), flush=True)


def train_model_graphnetwork_ltp_batch_allows_both(model, datasets,
                                 #dataloaders,
                                   criterion, optimizer, use_gpu, print_iter=10,
                save_iter=100, save_folder='/tmp',starting_epoch=0, final_epoch=1000, global_criterion=None,
                return_last_model_weights=True,dagger_train=False,train_env_name=None,seed=None,
                message_string='',
                log_wandb=False,
                ablation="main",
                chpkt_manager=None,
                enable_profiling=False,
                use_amp=False,
                spot_checkpoint_path=None,
                patience=0,
                domain_names=None):

    since = time.time()
    min_save_epoch = 0
    print_iter = 10
    save_iter = print_iter
    if use_gpu:
        model = model.cuda()
        device = "cuda:0"
        if criterion is not None:
            criterion = criterion.cuda()
    else:
        device = "cpu"

    device_type = 'cuda' if use_gpu else 'cpu'
    scaler = _GradScaler(enabled=use_amp and use_gpu)

    n_domains = len(domain_names) if domain_names else 0

    epochs = []
    train_loss_values = []
    val_loss_values = []
    time_taken_for_save_iter = time.time()
    avg_ranking_loss = None
    _best_val_for_patience = float('inf')
    _patience_counter = 0
    for epoch in range(starting_epoch,final_epoch+1):
        if epoch % print_iter == 0:
            print(f"Epoch {epoch}/{final_epoch}", end=" ", flush=True)
        # Each epoch has a training and validation phase
        running_num_samples = 0
        if epoch % print_iter == 0 :
            phases = ['train','val']
        else:
            phases = ['train']

        running_loss = {'train':0.0,'val':0.0}
        if n_domains > 0:
            domain_loss = {p: [0.0] * n_domains for p in phases}
            domain_count = {p: [0] * n_domains for p in phases}

        for phase in phases:
            if phase == 'train':
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()

            for i,batch_data in enumerate(datasets[phase]):
                optimizer.zero_grad()
                batch_data = batch_data.to(device)

                with torch.amp.autocast(device_type, enabled=use_amp):
                    if ablation == "main_val" :
                        (action_scores, action_object_scores), state_val = model.forward_with_value(batch_data)

                    elif ablation == "main" :
                        action_scores, action_object_scores = model(batch_data, beam_search=False)
                        #action_scores, action_object_scores = model.forward_with_parallel_beam_search(batch_data, beam_search=False)
                    tgt_action_scores = batch_data['target_action_scores'].x
                    tgt_action_object_scores = batch_data['target_action_object_scores'].x
                    tgt_params = batch_data['target_n_parameters'].x
                    # Old version (Python double loop per batch, with a CPU
                    # transfer per element of tgt_params):
                    #curr_param_counter = 0
                    #required_action_object_scores = []
                    #total_number_params = 0
                    #for idx,n_params in enumerate(tgt_params):
                    #    n_params = int(n_params)
                    #    for correct_index in range(curr_param_counter,curr_param_counter+n_params):
                    #        required_action_object_scores.append(correct_index)
                    #    curr_param_counter += model.max_number_action_parameters
                    #    total_number_params += n_params
                    #required_action_object_scores = torch.tensor(required_action_object_scores)
                    # Vectorized: identical indices, computed on-device
                    counts = tgt_params.long().flatten()
                    starts = torch.arange(counts.numel(), device=counts.device) * model.max_number_action_parameters
                    within = torch.arange(int(counts.sum()), device=counts.device) - \
                             torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
                    required_action_object_scores = torch.repeat_interleave(starts, counts) + within
                    target_indices = tgt_action_scores.argmax(dim=1)
                    target_indices_2 = tgt_action_object_scores[required_action_object_scores].argmax(dim=1)
                    tgt_action_scores = tgt_action_scores.squeeze(0)

                    m = torch.nn.ConstantPad2d((0,tgt_action_object_scores.shape[1]-action_object_scores.shape[1]\
                                                ,0,0),0)

                    action_object_scores = m(action_object_scores)
                    ranking_loss = 0.
                    ranking_loss += criterion(action_scores,target_indices)
                    ranking_loss += criterion(action_object_scores[required_action_object_scores],target_indices_2)

                    if ablation == "main_val" :
                        value_loss = compute_val_loss(state_val, batch_data)
                        alpha = 0.8
                        if avg_ranking_loss is None:
                            avg_ranking_loss = ranking_loss.item()
                            avg_value_loss = value_loss.item()
                        else:
                            avg_ranking_loss = 0.9 * avg_ranking_loss + 0.1 * ranking_loss.item()
                            avg_value_loss = 0.9 * avg_value_loss + 0.1 * value_loss.item()

                        # Use loss ratio to keep them in comparable ranges
                        loss_ratio = avg_ranking_loss / (avg_value_loss + 1e-8)

                        total_loss = alpha * ranking_loss + (1 - alpha) * (value_loss * loss_ratio)

                    else :
                        total_loss = ranking_loss

                if phase == 'train':
                    backward_time = time.time()
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    #ic ("backprop time",time.time()-backward_time)

                # statistics
                running_loss[phase] += total_loss.item()
                running_num_samples += 1

                if n_domains > 0 and hasattr(batch_data, 'domain_id'):
                    with torch.no_grad():
                        dids = batch_data.domain_id
                        if not isinstance(dids, torch.Tensor):
                            dids = torch.tensor(dids, dtype=torch.long)
                        dids = dids.long()
                        _as = action_scores.detach().float()
                        _ti = target_indices.detach()
                        # Object-selection rows are ordered graph-by-graph
                        # (counts = true arity per graph), so expanding the
                        # per-graph domain ids by counts aligns rows->domain.
                        _ao = action_object_scores[required_action_object_scores].detach().float()
                        _ti2 = target_indices_2.detach()
                        _row_dids = torch.repeat_interleave(dids, counts)
                        for d in dids.unique().tolist():
                            mask = (dids == d)
                            n = mask.sum().item()
                            domain_count[phase][d] += n
                            # action CE + object CE: the full ranking loss
                            # for this domain (a single-schema domain like
                            # Visitall has action CE == 0 by construction;
                            # its learning signal is entirely in objects).
                            _dl = torch.nn.functional.cross_entropy(
                                _as[mask], _ti[mask]).item()
                            rmask = (_row_dids == d)
                            if rmask.any():
                                _dl += torch.nn.functional.cross_entropy(
                                    _ao[rmask], _ti2[rmask]).item()
                            domain_loss[phase][d] += _dl * n

            if log_wandb:
                wandb.log({f"loss_{phase}": running_loss[phase]})

        if epoch % print_iter == 0:
            #print("running_loss:", running_loss, flush=True)
            print(f"loss: {running_loss} | time({save_iter}): {time.time() - time_taken_for_save_iter:.2f}s", flush=True)
            if n_domains > 0:
                parts = []
                for d in range(n_domains):
                    t_avg = domain_loss['train'][d] / max(1, domain_count['train'][d])
                    v_avg = domain_loss['val'][d] / max(1, domain_count['val'][d])
                    parts.append(f"{domain_names[d]}:t{t_avg:.3f}/v{v_avg:.3f}")
                print(f"  domain: {' | '.join(parts)}", flush=True)
            epochs.append(epoch)
            train_loss_values.append(running_loss['train'])
            val_loss_values.append(running_loss['val'])

            if patience > 0:
                if running_loss['val'] < _best_val_for_patience:
                    _best_val_for_patience = running_loss['val']
                    _patience_counter = 0
                else:
                    _patience_counter += 1
                    if _patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch} "
                              f"(no val improvement for {patience} checks)", flush=True)
                        break

        if epoch % save_iter == 0 and epoch >= min_save_epoch:
            chpkt_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_env_name=train_env_name,
                seed=42,
                losses={'train': running_loss["train"], 'val': running_loss["val"]},
            )
            #print ("Time taken for {} epochs : {}".format(save_iter, time.time() - time_taken_for_save_iter))
            time_taken_for_save_iter = time.time()

        if spot_checkpoint_path is not None:
            from ploi.run_modes import save_spot_checkpoint
            save_spot_checkpoint(spot_checkpoint_path, model, optimizer, epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60), flush=True)

# Modified training function with minimal code duplication
def train_model_graphnetwork_ltp_batch_profiling(model, datasets,
                                criterion, optimizer, use_gpu, print_iter=10,
                                save_iter=100, save_folder='/tmp', starting_epoch=0, final_epoch=1000,
                                global_criterion=None, return_last_model_weights=True, dagger_train=False,
                                train_env_name=None, seed=None, message_string='',
                                log_wandb=False,
                                ablation="main",
                                chpkt_manager=None,
                                enable_profiling=True,
                                use_amp=False,
                                spot_checkpoint_path=None,
                                patience=0,
                                domain_names=None,
                                profile_log_dir='./profile_logs'):

    since = time.time()
    min_save_epoch = 0
    save_iter = print_iter
    
    # Initialize the profiler manager if profiling is enabled
    profiler = ProfilerManager(log_dir=profile_log_dir) if enable_profiling else None
    
    if use_gpu:
        model = model.cuda()
        device = "cuda:0"
        if criterion is not None:
            criterion = criterion.cuda()
    else:
        device = "cpu"

    epochs = []
    train_loss_values = []
    val_loss_values = []
    time_taken_for_save_iter = time.time()
    
    for epoch in range(starting_epoch, final_epoch+1):
        epoch_start_time = time.time()
        
        # Determine if this epoch should have detailed output and profiling
        should_detail = epoch % print_iter == 0
        
        # Start profiling if enabled and this is a detail epoch
        if enable_profiling and should_detail:
            profiler.start_profiling_session(epoch)
        
        if should_detail:
            print('Epoch {}/{}'.format(epoch, final_epoch), flush=True)
            print('-' * 10, flush=True)
        
        # Each epoch has a training and validation phase
        running_num_samples = 0
        phases = ['train', 'val'] if should_detail else ['train']
        running_loss = {'train': 0.0, 'val': 0.0}

        for phase in phases:
            phase_start_time = time.time()
            
            # Set model mode based on phase
            model.train() if phase == 'train' else model.eval()

            for i, batch_data in enumerate(datasets[phase]):
                # Process batch with optional profiling
                with ConditionalRecordFunction(f"{phase}_batch", enable=enable_profiling and should_detail):
                    batch_start_time = time.time()
                    
                    optimizer.zero_grad()
                    batch_data = batch_data.to(device)
                    
                    # Forward pass
                    with ConditionalRecordFunction("forward", enable=enable_profiling and should_detail):
                        action_scores, action_object_scores = model(batch_data, beam_search=False)
                    
                    # Loss calculation
                    with ConditionalRecordFunction("loss_calculation", enable=enable_profiling and should_detail):
                        tgt_action_scores = batch_data['target_action_scores'].x
                        tgt_action_object_scores = batch_data['target_action_object_scores'].x
                        tgt_params = batch_data['target_n_parameters'].x
                        
                        loss = 0.
                        curr_param_counter = 0
                        required_action_object_scores = []
                        
                        for idx, n_params in enumerate(tgt_params):
                            n_params = int(n_params)
                            for correct_index in range(curr_param_counter, curr_param_counter+n_params):
                                required_action_object_scores.append(correct_index)
                            curr_param_counter += model.max_number_action_parameters
                        
                        required_action_object_scores = torch.tensor(required_action_object_scores)
                        target_indices = tgt_action_scores.argmax(dim=1)
                        target_indices_2 = tgt_action_object_scores[required_action_object_scores].argmax(dim=1)
                        tgt_action_scores = tgt_action_scores.squeeze(0)

                        m = torch.nn.ConstantPad2d((0, tgt_action_object_scores.shape[1]-action_object_scores.shape[1], 0, 0), 0)
                        action_object_scores = m(action_object_scores)
                        
                        loss += criterion(action_scores, target_indices)
                        loss += criterion(action_object_scores[required_action_object_scores], target_indices_2)

                    # Backward pass for training
                    if phase == 'train':
                        with ConditionalRecordFunction("backward", enable=enable_profiling and should_detail):
                            backward_time = time.time()
                            loss.backward()
                            optimizer.step()
                            backward_duration = time.time() - backward_time
                            
                            if should_detail and i == 0:
                                print(f"Backward pass time: {backward_duration:.4f}s")

                    # Statistics
                    running_loss[phase] += loss.item()
                    running_num_samples += 1
                    
                    # Print batch timing info for first few batches in detail epochs
                    if should_detail and i < 3:
                        batch_duration = time.time() - batch_start_time
                        print(f"  {phase} batch {i} time: {batch_duration:.4f}s")
                
                # Step the profiler after each batch if active
                if enable_profiling and should_detail:
                    profiler.step()
            
            # Print phase timing at detail epochs
            if should_detail:
                phase_duration = time.time() - phase_start_time
                print(f"  {phase} phase completed in {phase_duration:.2f}s")
            
            # Log to wandb if enabled
            if log_wandb:
                wandb.log({f"loss_{phase}": running_loss[phase]})

        # Print epoch summary at detail epochs
        if should_detail:
            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch {epoch} completed in {epoch_duration:.2f}s")
            print(f"Running loss: {running_loss}", flush=True)
            
            epochs.append(epoch)
            train_loss_values.append(running_loss['train'])
            val_loss_values.append(running_loss['val'])
            
            # End profiling session if active
            if enable_profiling:
                profiler.end_profiling_session()
    
        # Save checkpoint at save_iter intervals
        if epoch % save_iter == 0 and epoch >= min_save_epoch:
            checkpoint_start_time = time.time()
            chpkt_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_env_name=train_env_name,
                seed=42,
                losses={'train': running_loss["train"], 'val': running_loss["val"]},
            )
            checkpoint_duration = time.time() - checkpoint_start_time
            
            if should_detail:
                print(f"Checkpoint saved in {checkpoint_duration:.2f}s")
                print(f"Time taken for {save_iter} epochs: {time.time() - time_taken_for_save_iter:.2f}s")
            
            time_taken_for_save_iter = time.time()

    # Training complete
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60), flush=True)
    
    # Ensure profiler is stopped if active
    if enable_profiling and profiler and profiler.is_profiling:
        profiler.end_profiling_session()
        
    return model

def train_model_graphnetwork(
    model,
    datasets,
    criterion,
    optimizer,
    use_gpu=False,
    print_every=10,
    save_every=100,
    save_folder="/tmp",
    epochs=1000,
    global_criterion=None,
    return_last_model_weights=True,
):
    since = time.time()
    best_seen_model_weights = None  # as measured over the validation set
    best_seen_running_validation_loss = np.inf

    trainset, validset = datasets["train"], datasets["val"]

    if use_gpu:
        device = "cuda:0"
    else:
        device = "cpu"

    for e in range(epochs):

        running_loss = 0.0
        running_num_samples = 0

        model.train()

        for idx in range(len(trainset)):
            g_inp = trainset[idx]["graph_input"]
            g_tgt = trainset[idx]["graph_target"]
            nfeat = torch.from_numpy(g_inp["nodes"]).float().to(device)
            efeat = torch.from_numpy(g_inp["edges"]).float().to(device)
            senders = torch.from_numpy(g_inp["senders"]).long().to(device)
            receivers = torch.from_numpy(g_inp["receivers"]).long().to(device)
            tgt = torch.from_numpy(g_tgt["nodes"]).float().to(device)
            edge_indices = torch.stack((senders, receivers))
            preds = model(nfeat, edge_indices, efeat)
            loss = criterion(preds, tgt)

            running_loss += loss.item()
            running_num_samples += 1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(
            f"== [EPOCH {e:03d} / {epochs}] Train loss: {(running_loss / running_num_samples):03.5f}"
        )

        if e % 100 == 0:

            model.eval()

            if e % save_every == 0:
                savefile = os.path.join(save_folder, f"model_{e:04d}.pt")
                torch.save(model.state_dict(), savefile)
                print(f"Saved model checkpoint {savefile}")

            running_loss = 0.0
            running_num_samples = 0

            for idx in range(len(validset)):
                g_inp = validset[idx]["graph_input"]
                g_tgt = validset[idx]["graph_target"]
                nfeat = torch.from_numpy(g_inp["nodes"]).float().to(device)
                efeat = torch.from_numpy(g_inp["edges"]).float().to(device)
                senders = torch.from_numpy(g_inp["senders"]).long().to(device)
                receivers = torch.from_numpy(g_inp["receivers"]).long().to(device)
                tgt = torch.from_numpy(g_tgt["nodes"]).float().to(device)
                edge_indices = torch.stack((senders, receivers))
                preds = model(nfeat, edge_indices, efeat)
                loss = criterion(preds, tgt)

                running_loss += loss.item()
                running_num_samples += 1

            print(
                f"===== [EPOCH {e:03d} / {epochs}] Val loss: {(running_loss / running_num_samples):03.5f}"
            )

            val_loss = running_loss / running_num_samples
            if val_loss < best_seen_running_validation_loss:
                best_seen_running_validation_loss = copy.deepcopy(val_loss)
                best_seen_model_weights = model.state_dict()
                savefile = os.path.join(save_folder, "best.pt")
                torch.save(best_seen_model_weights, savefile)
                print(
                    f"Found new best model with val loss {best_seen_running_validation_loss} at epoch {e}. Saved!"
                )

    time_elapsed = time.time() - since
    print(
        f"Training complete in {(time_elapsed // 60):.0f} m {(time_elapsed % 60):.0f} sec"
    )

    return best_seen_model_weights

def predict_graph_with_graphnetwork(model, input_graph):
    """Predict the target graph given the input graph"""
    model.eval()
    nfeat = torch.from_numpy(input_graph["nodes"]).float()
    efeat = torch.from_numpy(input_graph["edges"]).float()
    senders = torch.from_numpy(input_graph["senders"]).long()
    receivers = torch.from_numpy(input_graph["receivers"]).long()
    edge_indices = torch.stack((senders, receivers))
    scores = model(nfeat, edge_indices, efeat)
    scores = torch.sigmoid(scores)
    input_graph["nodes"] = scores.detach().cpu().numpy()
    return input_graph

def test_planner(
    planner, domain_name, num_problems, timeout, debug_mode=False, all_problems=False
):
    print("Running testing...")
    # In debug mode, use train problems for testing too (False by default)
    env = pddlgym.make("PDDLEnv{}Test-v0".format(domain_name))
    if debug_mode:
        warnings.warn(
            "WARNING: Running in debug mode (i.e., testing on train problems)"
        )
        env = pddlgym.make("PDDLEnv{}-v0".format(domain_name))
    num_problems = min(num_problems, len(env.problems))
    # If `all_problems` is set to True, override num_problems
    if all_problems:
        num_problems = len(env.problems)
    stats_to_log = ["num_node_expansions", "plan_length", "search_time", "total_time"]
    num_timeouts = 0
    num_failures = 0
    num_invalidated_plans = 0
    run_stats = []
    for problem_idx in range(num_problems):
        print(
            "\tTesting problem {} of {}".format(problem_idx + 1, num_problems),
            flush=True,
        )
        env.fix_problem_index(problem_idx)
        state, _ = env.reset()
        start = time.time()
        try:
            plan, planner_stats = planner(
                env.domain, state, timeout=timeout, 
                #domain_file_global=None# 
                domain_file_global=env._domain_file
            )
        except PlanningFailure as e:
            num_failures += 1
            print("\t\tPlanning failed with error: {}".format(e), flush=True)
            continue
        except PlanningTimeout as e:
            num_timeouts += 1
            print("\t\tPlanning failed with error: {}".format(e), flush=True)
            continue
        # Validate plan on the full test problem.
        if plan is None:
            num_failures += 1
            continue
        if not validate_strips_plan(
            domain_file=env.domain.domain_fname,
            problem_file=env.problems[problem_idx].problem_fname,
            plan=plan,
        ):
            print("\t\tPlanning returned an invalid plan")
            num_invalidated_plans += 1
            continue
        wall_time = time.time() - start
        print(
            "\t\tSuccess, got plan of length {} in {:.5f} seconds".format(
                len(plan), wall_time
            ),
            flush=True,
        )
        planner_stats["wall_time"] = wall_time
        run_stats.append(planner_stats)

    global_stats = dict()
    stats_to_track = {
        "num_node_expansions",
        "plan_length",
        "search_time",
        "total_time",
        "objects_used",
        "objects_total",
        "neural_net_time",
        "wall_time",
    }
    num_stats = len(run_stats)
    for stat in stats_to_track:
        if stat not in global_stats:
            global_stats[stat] = np.zeros(num_stats)
        for i, run in enumerate(run_stats):
            global_stats[stat][i] = run[stat]
    for stat in stats_to_track:
        stat_mean = float(global_stats[stat].mean().item())
        stat_std = float(global_stats[stat].std().item())
        global_stats[stat] = stat_mean
        global_stats[f"{stat}_std"] = stat_std
    global_stats["success_rate"] = float(num_stats / num_problems)
    global_stats["timeout_rate"] = float(num_timeouts / num_problems)
    global_stats["failure_rate"] = float(num_failures / num_problems)
    global_stats["invalid_rate"] = float(num_invalidated_plans / num_problems)

    global_stats["num_timeouts"] = num_timeouts
    global_stats["num_failures"] = num_failures
    global_stats["num_invalidated_plans"] = num_invalidated_plans
    global_stats["num_timeouts"] = num_timeouts
    return run_stats, global_stats

def train_model_hierarchical(
    model,
    datasets,
    criterion,
    optimizer,
    use_gpu=False,
    print_every=10,
    save_every=100,
    save_folder="/tmp",
    epochs=1000,
    global_criterion=None,
    return_last_model_weights=True,
    model_type="room",
    eval_every=100,
):
    if model_type not in ["room", "object"]:
        raise ValueError(
            f"Unknown model type {model_type}. Valid model types are 'room', 'object'."
        )

    if use_gpu:
        device = "cuda:0"
    else:
        device = "cpu"

    def unpack_item(item):
        if model_type == "object":
            _input_graph = item["graph_input"]
            _target_graph = item["graph_target"]
            _nfeat = _input_graph["nodes"].float().to(device)
            _efeat = _input_graph["edges"].float().to(device)
            _senders = _input_graph["senders"].long().to(device)
            _receivers = _input_graph["receivers"].long().to(device)
            _tgt = _target_graph["nodes"].float().to(device)
            _edge_indices = torch.stack((_senders, _receivers))
            return _nfeat, _edge_indices, _efeat, _tgt
        elif model_type == "room":
            _input_graph = item["graph_input"]
            _target_graph = item["graph_target"]
            _nfeat = _input_graph["room_graph"]["nodes"].float().to(device)
            _efeat = _input_graph["room_graph"]["edges"].float().to(device)
            _senders = _input_graph["room_graph"]["senders"].long().to(device)
            _receivers = _input_graph["room_graph"]["receivers"].long().to(device)
            _tgt = _target_graph["room_graph"]["nodes"].float().to(device)
            _edge_indices = torch.stack((_senders, _receivers))
            return _nfeat, _edge_indices, _efeat, _tgt

    since = time.time()
    best_seen_model_weights = None  # as measured over the validation set
    best_seen_running_validation_loss = np.inf

    trainset, validset = datasets["train"], datasets["val"]

    for e in range(epochs):

        running_loss = 0.0
        running_num_samples = 0

        model.train()

        permuted_indx = torch.randperm(len(trainset))

        for idx in range(len(trainset)):
            i = permuted_indx[idx]
            nfeat, edge_indices, efeat, tgt = unpack_item(trainset[i])
            preds = model(nfeat, edge_indices, efeat)
            loss = criterion(preds, tgt)

            running_loss += loss.item()
            running_num_samples += 1

            loss.backward()

            if idx % 20 == 0 or idx == len(trainset) - 1:
                optimizer.step()
                optimizer.zero_grad()

        print(
            f"== [Model: {model_type}] [EPOCH {e:03d} / {epochs}] Train loss: {(running_loss / running_num_samples):03.5f}"
        )

        if e % eval_every == 0:

            model.eval()

            if e % save_every == 0:
                savefile = os.path.join(save_folder, f"{model_type}_model_{e:04d}.pt")
                torch.save(model.state_dict(), savefile)
                print(f"Saved model checkpoint {savefile}")

            running_loss = 0.0
            running_num_samples = 0

            for idx in range(len(validset)):
                nfeat, edge_indices, efeat, tgt = unpack_item(validset[idx])
                preds = model(nfeat, edge_indices, efeat)
                loss = criterion(preds, tgt)

                running_loss += loss.item()
                running_num_samples += 1

            print(
                f"===== [Model: {model_type}] [EPOCH {e:03d} / {epochs}] Val loss: {(running_loss / running_num_samples):03.5f}"
            )

            val_loss = running_loss / running_num_samples
            if val_loss < best_seen_running_validation_loss:
                best_seen_running_validation_loss = copy.deepcopy(val_loss)
                best_seen_model_weights = model.state_dict()
                savefile = os.path.join(save_folder, f"{model_type}_best.pt")
                torch.save(best_seen_model_weights, savefile)
                print(
                    f"Found new best model with val loss {best_seen_running_validation_loss} at epoch {e}. Saved!"
                )

    time_elapsed = time.time() - since
    print(
        f"Training complete in {(time_elapsed // 60):.0f} m {(time_elapsed % 60):.0f} sec"
    )

    return best_seen_model_weights


def predict_graph_with_graphnetwork_hierarchical(room_model, object_model, input_graph):
    """Predict scores across both levels of the hierarchy. """
    room_model.eval()
    object_model.eval()

    # Get room scores
    nfeat = torch.from_numpy(input_graph["room_graph"]["nodes"]).float()
    efeat = torch.from_numpy(input_graph["room_graph"]["edges"]).float()
    senders = torch.from_numpy(input_graph["room_graph"]["senders"]).long()
    receivers = torch.from_numpy(input_graph["room_graph"]["receivers"]).long()
    edge_indices = torch.stack((senders, receivers))
    room_scores = room_model(nfeat, edge_indices, efeat)
    room_scores = torch.sigmoid(room_scores)

    nfeat = torch.from_numpy(input_graph["nodes"]).float()
    efeat = torch.from_numpy(input_graph["edges"]).float()
    senders = torch.from_numpy(input_graph["senders"]).long()
    receivers = torch.from_numpy(input_graph["receivers"]).long()
    edge_indices = torch.stack((senders, receivers))
    object_scores = object_model(nfeat, edge_indices, efeat)
    object_scores = torch.sigmoid(object_scores)

    return room_scores, object_scores
