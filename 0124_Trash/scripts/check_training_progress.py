#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check training progress and estimate completion time for all running processes
"""

import os
import re
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import json

def get_process_info(pid):
    """Get process information"""
    try:
        result = subprocess.run(['ps', '-p', str(pid), '-o', 'etime,cmd', '--no-headers'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return None

def extract_epoch_from_log(log_file):
    """Extract current epoch from log file"""
    if not os.path.exists(log_file):
        return None
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        # Look for epoch information in the last 100 lines
        for line in reversed(lines[-100:]):
            # Match patterns like "Epoch X/Y" or "Epoch X/Y:"
            epoch_match = re.search(r'Epoch\s+(\d+)/(\d+)', line, re.IGNORECASE)
            if epoch_match:
                current = int(epoch_match.group(1))
                total = int(epoch_match.group(2))
                return current, total
            
            # Match completion messages
            if '训练完成' in line or 'Training completed' in line.lower() or 'completed' in line.lower():
                return None, None  # Training is done
                
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
    
    return None

def extract_time_per_epoch(log_file):
    """Extract average time per epoch from log"""
    if not os.path.exists(log_file):
        return None
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Look for "Epoch Time" patterns
        time_patterns = [
            r'Epoch Time:\s*(\d+\.?\d*)\s*s',
            r'Epoch Time:\s*(\d+\.?\d*)\s*seconds',
            r'epoch.*time.*?(\d+\.?\d*)\s*s',
        ]
        
        times = []
        for pattern in time_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                times.extend([float(t) for t in matches])
        
        if times:
            return sum(times) / len(times)  # Average time per epoch
            
    except:
        pass
    
    return None

def estimate_completion(pid, log_file, epochs_config=20):
    """Estimate when training will complete"""
    info = {}
    
    # Get process elapsed time
    proc_info = get_process_info(pid)
    if proc_info:
        info['process_info'] = proc_info
        # Extract elapsed time (format: DD-HH:MM:SS or HH:MM:SS)
        elapsed_match = re.search(r'(\d+-)?(\d+):(\d+):(\d+)', proc_info)
        if elapsed_match:
            days = int(elapsed_match.group(1).rstrip('-')) if elapsed_match.group(1) else 0
            hours = int(elapsed_match.group(2))
            minutes = int(elapsed_match.group(3))
            seconds = int(elapsed_match.group(4))
            total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
            info['elapsed_seconds'] = total_seconds
    else:
        info['status'] = 'not_found'
        return info
    
    # Get current epoch from log
    epoch_info = extract_epoch_from_log(log_file)
    if epoch_info:
        current_epoch, total_epochs = epoch_info
        if current_epoch is None:
            info['status'] = 'completed'
            return info
        
        info['current_epoch'] = current_epoch
        info['total_epochs'] = total_epochs
        remaining_epochs = total_epochs - current_epoch
        
        # Get average time per epoch
        avg_time = extract_time_per_epoch(log_file)
        if avg_time:
            info['avg_seconds_per_epoch'] = avg_time
            remaining_seconds = remaining_epochs * avg_time
            info['estimated_remaining_seconds'] = remaining_seconds
            info['estimated_completion'] = datetime.now() + timedelta(seconds=remaining_seconds)
        else:
            # Estimate based on elapsed time
            if 'elapsed_seconds' in info and current_epoch > 0:
                avg_time = info['elapsed_seconds'] / current_epoch
                info['avg_seconds_per_epoch'] = avg_time
                remaining_seconds = remaining_epochs * avg_time
                info['estimated_remaining_seconds'] = remaining_seconds
                info['estimated_completion'] = datetime.now() + timedelta(seconds=remaining_seconds)
    else:
        # Try to get from metrics.json
        metrics_file = log_file.replace('train.log', 'metrics.json')
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    current = metrics.get('epoch', 0)
                    if current > 0:
                        info['current_epoch'] = current
                        info['total_epochs'] = epochs_config
                        remaining_epochs = epochs_config - current
                        if 'elapsed_seconds' in info and current > 0:
                            avg_time = info['elapsed_seconds'] / current
                            info['avg_seconds_per_epoch'] = avg_time
                            remaining_seconds = remaining_epochs * avg_time
                            info['estimated_remaining_seconds'] = remaining_seconds
                            info['estimated_completion'] = datetime.now() + timedelta(seconds=remaining_seconds)
            except:
                pass
    
    return info

def main():
    """Main function"""
    base_dir = Path('/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713')
    
    # Process mapping: PID -> (log_file, epochs_config, method_name)
    processes = {
        110016: (base_dir / 'cnn_result_enhanced' / 'train.log', 20, 'CNN Enhanced'),
        4126863: (base_dir / 'vmamba_result' / 'train.log', 20, 'Vmamba'),
        3832686: (base_dir / 'cnn_result_large' / 'train.log', 20, 'CNN Large'),
        4126982: (base_dir / 'cnn_result' / 'train.log', 20, 'CNN'),
    }
    
    print("="*80)
    print("Training Progress and Estimated Completion Time")
    print("="*80)
    print()
    
    for pid, (log_file, epochs, method) in processes.items():
        print(f"Method: {method} (PID: {pid})")
        print(f"Log file: {log_file}")
        print(f"Configured epochs: {epochs}")
        print("-" * 80)
        
        info = estimate_completion(pid, str(log_file), epochs)
        
        if 'status' in info:
            if info['status'] == 'completed':
                print("Status: Training completed!")
            elif info['status'] == 'not_found':
                print("Status: Process not found (may have finished)")
        else:
            if 'current_epoch' in info:
                print(f"Progress: Epoch {info['current_epoch']}/{info['total_epochs']}")
                if 'avg_seconds_per_epoch' in info:
                    print(f"Average time per epoch: {info['avg_seconds_per_epoch']:.1f} seconds ({info['avg_seconds_per_epoch']/60:.1f} minutes)")
                
                if 'estimated_completion' in info:
                    remaining = info['estimated_remaining_seconds']
                    hours = int(remaining // 3600)
                    minutes = int((remaining % 3600) // 60)
                    print(f"Estimated remaining time: {hours}h {minutes}m")
                    print(f"Estimated completion: {info['estimated_completion'].strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print("Status: Training in progress (epoch info not available)")
            
            if 'elapsed_seconds' in info:
                elapsed = info['elapsed_seconds']
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                print(f"Elapsed time: {hours}h {minutes}m")
        
        print()
    
    # Check SwinT status
    swint_log = base_dir / 'models' / 'SwinT' / '_results' / 'multimodal' / 'train.log'
    if swint_log.exists():
        print("Method: SwinT")
        print(f"Log file: {swint_log}")
        print("-" * 80)
        
        # Check for errors
        with open(swint_log, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            if 'OutOfMemoryError' in content or 'CUDA out of memory' in content:
                print("Status: ❌ CUDA Out of Memory Error - Training failed")
                print("Recommendation: Reduce batch size or use gradient accumulation")
            else:
                epoch_info = extract_epoch_from_log(str(swint_log))
                if epoch_info:
                    current, total = epoch_info
                    if current is None:
                        print("Status: Training completed!")
                    else:
                        print(f"Progress: Epoch {current}/{total}")
                else:
                    print("Status: Training in progress or failed")
        print()

if __name__ == '__main__':
    main()

