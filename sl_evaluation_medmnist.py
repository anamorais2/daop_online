import json
from logging import config
import numpy as np
import torch
import sys
import os
import torch.nn.functional as F
from sklearn.metrics import (f1_score, balanced_accuracy_score, precision_score, 
                             recall_score, matthews_corrcoef, confusion_matrix, roc_auc_score, roc_curve)
import analysis.utils as utils
import pandas as pd

DATASET_INFO = {
            'breastmnist': {
                'classes': ['Malignant', 'Benign'],
                'title': 'BreastMNIST'
            },
            'pneumoniamnist': {
                'classes': ['Normal', 'Pneumonia'],
                'title': 'PneumoniaMNIST'
            },
            'dermamnist': {
                'classes': ['AKIE', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC'],
                'title': 'DermaMNIST'
            },
            'organcmnist': {
                'classes': ['Bladder', 'Femur-L', 'Femur-R', 'Heart', 'Kidney-L', 'Kidney-R', 'Liver', 'Lung-L', 'Lung-R', 'Pancreas', 'Spleen'],
                'title': 'OrganCMNIST'
            }
        }


def train_sl(model, trainloader, config):
    device = config['device']
    model.model.to(device)
    model.model.train()
    
    criterion = model.criterion() 
    optimizer = model.optimizer(model.model.parameters()) 

    hist_loss, hist_acc, hist_wf1, hist_auc, hist_bal_acc = [], [], [], [], []
    
    last_epoch_targets, last_epoch_preds = [], []

    for epoch in range(config['epochs']):
        running_loss = 0
        correct = 0
        total = 0
        epoch_labels, epoch_preds, epoch_probs = [], [], []
        
        for batch in trainloader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            
            labels = batch["label"]
            
            optimizer.zero_grad()
            
        
            outputs = model.model(batch)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            epoch_labels.extend(labels.cpu().numpy())
            epoch_preds.extend(preds.cpu().numpy())
            epoch_probs.extend(F.softmax(outputs, dim=1).detach().cpu().numpy())
        
        final_loss = running_loss / len(trainloader)
        final_acc = correct / total 
        final_wf1 = f1_score(epoch_labels, epoch_preds, average='weighted')
        final_bal_acc = balanced_accuracy_score(epoch_labels, epoch_preds)
        
        try:
            probs_pos = [p[1] for p in epoch_probs]
            final_auc = roc_auc_score(epoch_labels, probs_pos)
        except:
            final_auc = 0.0

        print(f"Epoch {epoch + 1}/{config['epochs']} | Loss: {final_loss:.4f} | "
              f"Acc: {final_acc:.4f} | WF1: {final_wf1:.4f}")
        
        hist_loss.append(final_loss)
        hist_acc.append(final_acc)
        hist_wf1.append(final_wf1)
        hist_auc.append(final_auc)
        hist_bal_acc.append(final_bal_acc)
        
        last_epoch_targets = epoch_labels
        last_epoch_preds = epoch_preds


        #train_conf_matrix = confusion_matrix(last_epoch_targets, last_epoch_preds, labels=[1, 0])
        #print("Final Train Confusion Matrix:\n", train_conf_matrix)

    print("Finished Multimodal SL training")    
    return hist_loss, hist_acc, hist_wf1, hist_bal_acc, hist_auc

def train_sl_EML(model, trainloader, config):
    device = config['device']
    model.model.to(device)
    model.model.train()
    
    criterion = model.criterion() 
    # Note: We keep the same optimizer across epochs to allow for incremental updates without reinitializing
    if not hasattr(model, 'current_optimizer') or model.current_optimizer is None:
        model.current_optimizer = model.optimizer(model.model.parameters())
    
    optimizer = model.current_optimizer
    
    print(f"DEBUG: Optimizer ID: {id(optimizer)}")

    hist_loss, hist_acc, hist_wf1, hist_auc, hist_bal_acc = [], [], [], [], []
    
    last_epoch_targets, last_epoch_preds = [], []

    for epoch in range(config['epochs']):
        running_loss = 0
        correct = 0
        total = 0
        epoch_labels, epoch_preds, epoch_probs = [], [], []
        
        for batch in trainloader:
            #for k in batch:
            #    if isinstance(batch[k], torch.Tensor):
            #        batch[k] = batch[k].to(device)
            
            images = batch["images"].to(device)
            
            labels = batch["label"].to(device)
        
            
            optimizer.zero_grad()
            
            outputs = model.model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            epoch_labels.extend(labels.cpu().numpy())
            epoch_preds.extend(preds.cpu().numpy())
            epoch_probs.extend(F.softmax(outputs, dim=1).detach().cpu().numpy())
        
        final_loss = running_loss / len(trainloader)
        final_acc = correct / total 
        final_wf1 = f1_score(epoch_labels, epoch_preds, average='weighted')
        final_bal_acc = balanced_accuracy_score(epoch_labels, epoch_preds)
        
        try:
            probs_pos = [p[1] for p in epoch_probs]
            final_auc = roc_auc_score(epoch_labels, probs_pos)
        except:
            final_auc = 0.0

        print(f"Epoch {epoch + 1}/{config['epochs']} | Loss: {final_loss:.4f} | "
              f"Acc: {final_acc:.4f} | WF1: {final_wf1:.4f}")
        
        hist_loss.append(final_loss)
        hist_acc.append(final_acc)
        hist_wf1.append(final_wf1)
        hist_auc.append(final_auc)
        hist_bal_acc.append(final_bal_acc)
        
        last_epoch_targets = epoch_labels
        last_epoch_preds = epoch_preds


        #train_conf_matrix = confusion_matrix(last_epoch_targets, last_epoch_preds, labels=[1, 0])
        #print("Final Train Confusion Matrix:\n", train_conf_matrix)

    print("Finished Multimodal SL training")    
    return hist_loss, hist_acc, hist_wf1, hist_bal_acc, hist_auc
    

def train_sl_incremental_dynamic(model, trainloader, suploader, config):
    device = config['device']
    model.model.to(device)
    model.model.train() 
    
    criterion = model.criterion()
    optimizer = model.optimizer(model.model.parameters())
    
    warmup_epochs = config.get('warmup_epochs', 20) 
    num_intervals = config.get('num_intervals', 5)
    epochs_per_interval = config.get('epochs_per_interval', 10)

    hist_loss, hist_acc, hist_wf1, hist_auc, hist_bal_acc = [], [], [], [], []
    
    # --- PHASE 1: WARMUP ---
    print("\n  [INFO] PHASE 1: Warmup Training (Standard) for {} epochs".format(warmup_epochs))
    for epoch in range(warmup_epochs):
        running_loss, correct, total = 0.0, 0, 0
        epoch_labels, epoch_preds, epoch_probs = [], [], []
        
        for batch in trainloader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            
            labels = batch["label"]
            optimizer.zero_grad()
            outputs = model.model(batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            epoch_labels.extend(labels.cpu().numpy())
            epoch_preds.extend(preds.cpu().numpy())
            epoch_probs.extend(F.softmax(outputs, dim=1).detach().cpu().numpy())
        
        avg_loss = running_loss / len(trainloader)
        avg_acc = correct / total
        avg_wf1 = f1_score(epoch_labels, epoch_preds, average='weighted', zero_division=0)
        avg_bal_acc = balanced_accuracy_score(epoch_labels, epoch_preds)
        
        try:
            avg_auc = roc_auc_score(epoch_labels, [p[1] for p in epoch_probs])
        except:
            avg_auc = 0.0
        
        hist_loss.append(avg_loss)
        hist_acc.append(avg_acc)
        hist_wf1.append(avg_wf1)
        hist_bal_acc.append(avg_bal_acc)
        hist_auc.append(avg_auc)
        
        print(f"Warmup Epoch {epoch+1}/{warmup_epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | WF1: {avg_wf1:.4f}")

    # --- PHASE 2: DAOP DYNAMIC TRAINING ---
    print("\n  [INFO] PHASE 2: Dynamic Augmented Intervals (DAOP)")
    
    for interval in range(num_intervals):
        print(f"\n>> Interval {interval + 1}/{num_intervals}")
        
        for epoch in range(epochs_per_interval):
            interval_loss, int_correct, int_total = 0.0, 0, 0
            interval_labels, interval_preds, interval_probs = [], [], []
        
            for part_batch in suploader:
                for k in part_batch:
                    if isinstance(part_batch[k], torch.Tensor):
                        part_batch[k] = part_batch[k].to(device)
                
                labels = part_batch["label"]
                optimizer.zero_grad()
                outputs = model.model(part_batch)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                interval_loss += loss.item()
                preds = outputs.argmax(dim=1)
                int_correct += (preds == labels).sum().item()
                int_total += labels.size(0)
                
                interval_labels.extend(labels.cpu().numpy())
                interval_preds.extend(preds.cpu().numpy())
                interval_probs.extend(F.softmax(outputs, dim=1).detach().cpu().numpy())

            avg_loss_int = interval_loss / (len(suploader) + 1e-6)
            avg_acc_int = int_correct / int_total
            avg_wf1_int = f1_score(interval_labels, interval_preds, average='weighted', zero_division=0)
            
            hist_loss.append(avg_loss_int)
            hist_acc.append(avg_acc_int)
            hist_wf1.append(avg_wf1_int)
            hist_bal_acc.append(balanced_accuracy_score(interval_labels, interval_preds))
            try:
                hist_auc.append(roc_auc_score(interval_labels, [p[1] for p in interval_probs]))
            except:
                hist_auc.append(0.0)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"   [Interval {interval+1}] Epoch {epoch+1}/{epochs_per_interval} | Loss: {avg_loss_int:.4f} | Acc: {avg_acc_int:.4f} | WF1: {avg_wf1_int:.4f}")

    print("\n[INFO] Finished DAOP Dynamic Training")
    return hist_loss, hist_acc, hist_wf1, hist_bal_acc, hist_auc


def run_inference(model, loader, device):
    model.model.to(device)
    model.model.eval()

    predictions = []
    targets = []
    prob_scores = []

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            labels = batch["label"].to(device)
            
            if labels.dim() > 1 and labels.shape[1] == 1:
                labels = labels.squeeze(1)

            outputs = model.model(images)
            preds = outputs.argmax(dim=1)
            
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            prob_scores.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())

    targets = np.array(targets)
    predictions = np.array(predictions)
    prob_scores = np.array(prob_scores)

    unique_classes = np.unique(targets)
    num_classes = len(unique_classes)

    balanced_accuracy = balanced_accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='weighted', zero_division=1)
    recall = recall_score(targets, predictions, average='weighted')
    f1 = f1_score(targets, predictions, average='weighted')
    matthews = matthews_corrcoef(targets, predictions)
    
    test_conf_matrix = confusion_matrix(targets, predictions) 
    
    if num_classes > 2:
        roc_auc = roc_auc_score(targets, prob_scores, multi_class='ovr', average='macro')
    else:
        roc_auc = roc_auc_score(targets, prob_scores[:, 1])

  
    acc = (predictions == targets).mean()

    results = {
        "acc": acc,
        "bal_acc": balanced_accuracy, 
        "precision": precision,
        "recall": recall,
        "wf1": f1, 
        "matthews": matthews,
        "roc_auc": roc_auc,
        "confusion_matrix": test_conf_matrix,
        "all_labels": targets,
        "all_preds": predictions,
        "all_probs": prob_scores
    }

    if num_classes == 2:
        tn, fp, fn, tp = test_conf_matrix.ravel()
        results.update({
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "tpr": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "tnr": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0
        })
    else:
        #For multi-class, we can set these to None or 0 since they don't apply globally
        results.update({"tp": 0, "fn": 0, "fp": 0, "tn": 0, "tpr": recall, "tnr": 0, "specificity": 0})

    return results


def evaluate_sl(trainloader, valloader, testloader, config, model=None):
    
    if model is None:
        model = config['model'](config)
    
    if config.get('online_training', False):
        print("\n  [INFO] Starting SL training with EML")
        sl_hist_loss, sl_hist_acc, sl_hist_wf1, sl_hist_bal, sl_hist_auc = train_sl_EML(model, trainloader, config=config)
    else:
        print("\n  [INFO] Starting SL training with Standard setup...")
        sl_hist_loss, sl_hist_acc, sl_hist_wf1, sl_hist_bal, sl_hist_auc = train_sl(model, trainloader, config)
    
    val_results = run_inference(model, valloader, config['device'])

    test_results = None
    if config.get('evaluate_on_test', False): 
        
        print("\n  [INFO] Performing Final Evaluation")
        test_results = run_inference(model, testloader, config['device'])
        
        ds_name = config.get('dataset', 'unknown').lower()
        info = DATASET_INFO.get(ds_name, {
            'classes': [str(i) for i in range(test_results['confusion_matrix'].shape[0])],
            'title': ds_name
        })

        base_res_dir = config.get('output_csv_folder', 'final_results')        
        res_dir = os.path.join(base_res_dir, f"seed_{config.get('seed', -1)}")
        os.makedirs(res_dir, exist_ok=True)

        model_path = os.path.join(res_dir, f"final_model_seed{config.get('seed', 0)}.pth")
        torch.save(model.model.state_dict(), model_path)
        print(f"Weights saved to {model_path}")

        config_path = os.path.join(res_dir, "config_model.json")
        final_config_json = {
            "lr": config.get('lr'), 
            "batch_size": config.get('batch_size'),
            "epochs": config.get('epochs'), 
            "stop_epochs": config.get('stop_epochs'),
            "generation": config.get('stop_gen', -1),
            "warmup_epochs": config.get('warmup_epochs', 0),
            "gens_per_epoch": config.get('gens_per_epoch', 0),
            "num_numerical": config.get('num_numerical'),
            "num_categories": config.get('num_categories')
        }
        with open(config_path, 'w') as f:
            json.dump(final_config_json, f, indent=4)
            
        utils.save_test_metrics(
            os.path.join(res_dir, "test_metrics.json"),
            balanced_accuracy=test_results['bal_acc'],
            acc=test_results['acc'],
            precision=test_results['precision'],
            recall=test_results['recall'],
            f1=test_results['wf1'],
            roc_auc=test_results['roc_auc'],
            matthews=test_results.get('matthews', 0),
            specificity=test_results.get('specificity', 0),
            tnr=test_results.get('tnr', 0)
        )
        
        utils.plot_confusionMatrix(
            os.path.join(res_dir, "confusion_matrix.png"),
            test_results['confusion_matrix'],
            class_names=info['classes'],
            title_fig="Confusion Matrix" + config['dataset']
        )
        utils.plot_RocCurve_both(
            os.path.join(res_dir, "roc_curve.png"),
            targets=test_results['all_labels'],
            prob_scores=test_results['all_probs'], 
            class_names = info['classes']
        )
        
        
        """
        test_df = testloader.dataset.df
        probs = np.array(test_results['all_probs']) 
        pred_df = pd.DataFrame({
            "Processo": test_df["Processo"].reset_index(drop=True),
            "y_true": test_results['all_labels'],
            "y_pred": test_results['all_preds'],
            "y_proba_0": probs[:, 0],
            "y_proba_1": probs[:, 1], 
        })
        
        pred_df.to_csv(os.path.join(res_dir, "predictions.csv"), index=False)
        print(f"Predictions saved to {os.path.join(res_dir, 'predictions.csv')}")
        """   
        print("\nTraining and evaluation completed successfully!")
        print(f"Results folder: {res_dir}")

    if test_results:
        print(f"Test Results: Acc: {test_results['acc']:.4f} | BalAcc: {test_results['bal_acc']:.4f} | WF1: {test_results['wf1']:.4f} | AUC: {test_results['roc_auc']:.4f}")
    else:
        print(f"VAL: Acc: {val_results['acc']:.4f} | BalAcc: {val_results['bal_acc']:.4f} | WF1: {val_results['wf1']:.4f} | AUC: {val_results['roc_auc']:.4f}")

    history = {
        "sl_hist_loss": sl_hist_loss,
        "sl_hist_acc": sl_hist_acc,
        "sl_hist_wf1": sl_hist_wf1,
        "sl_hist_bal_acc": sl_hist_bal,
        "sl_hist_auc": sl_hist_auc,

        "val_acc": val_results['acc'],
        "val_bal_acc": val_results['bal_acc'],
        "val_wf1": val_results['wf1'],
        "val_auc": val_results['roc_auc'],
        "val_precision": val_results['precision'],
        "val_recall": val_results['recall'],
        "val_matthews": val_results['matthews'],
        "val_specificity": val_results['specificity'],
        "val_confusion_matrix": val_results['confusion_matrix'],

    }
    
    return val_results['acc'], history

def evaluate_only_test_set(testloader, config, model_path=None):
    print("\n  [INFO] Performing Pure Inference (No Training)...")
    
    model = config['model'](config)
    device = config['device']
    seed = config.get('seed', 0)
    
    base_res_dir = config.get('output_csv_folder', 'final_results')
    res_dir = os.path.join(base_res_dir, f"seed_{seed}")
    os.makedirs(res_dir, exist_ok=True)

    if model_path is None:
        model_path = os.path.join(res_dir, f"final_model_seed{config.get('seed', 0)}.pth")

    if not os.path.exists(model_path):
        print(f" [ERROR] Model file not found at {model_path}")
        return None

    model.model.load_state_dict(torch.load(model_path, map_location=device))
    print(f" [INFO] Weights loaded successfully from {model_path}")

    test_results = run_inference(model, testloader, device)

    
    config_path = os.path.join(res_dir, f"config_model_inference_seed_{seed}.json")
    final_config_json = {
        "lr": config.get('lr'), 
        "batch_size": config.get('batch_size'),
        "num_numerical": config.get('num_numerical'),
        "num_categories": config.get('num_categories'),
        "inference_only": True
    }
    with open(config_path, 'w') as f:
        json.dump(final_config_json, f, indent=4)

    """
    utils.plot_confusionMatrix(
        os.path.join(res_dir, f"confusion_matrix_inference_seed_{seed}.png"),
        test_results['confusion_matrix'],
        class_names=['Cesarean birth', 'Vaginal birth'],
        title_fig="Confusion Matrix (Prospective Evaluation - Inference Only)"
    )
    
    utils.plot_RocCurve(
        os.path.join(res_dir, f"roc_curve_inference_seed_{seed}.png"),
        targets=test_results['all_labels'],
        prob_scores=test_results['all_probs']
    )
"""
    utils.save_test_metrics(
        os.path.join(res_dir, f"test_metrics_inference_seed_{seed}.json"),
        balanced_accuracy=test_results['bal_acc'],
        acc=test_results['acc'],
        precision=test_results['precision'],
        recall=test_results['recall'],
        f1=test_results['wf1'],
        roc_auc=test_results['roc_auc'],
        matthews=test_results.get('matthews', 0),
        specificity=test_results.get('specificity', 0),
        tnr=test_results.get('tnr', 0)
    )
    
    """
    test_df = testloader.dataset.df
    probs = np.array(test_results['all_probs']) 
    pred_df = pd.DataFrame({
        "Processo": test_df["Processo"].reset_index(drop=True),
        "y_true": test_results['all_labels'],
        "y_pred": test_results['all_preds'],
        "y_proba_0": probs[:, 0],
        "y_proba_1": probs[:, 1], 
    })
    
    csv_path = os.path.join(res_dir, "predictions_inference.csv")
    pred_df.to_csv(csv_path, index=False)
    """
    print(f"\n[SUCCESS] Inference completed!")
    print(f"Results saved in: {res_dir}")
    print(f"Test Results: Acc: {test_results['acc']:.4f} | BalAcc: {test_results['bal_acc']:.4f} | WF1: {test_results['wf1']:.4f} | AUC: {test_results['roc_auc']:.4f}")

    return test_results