import os
import itertools
import subprocess
import time
import argparse
from typing import List, Dict, Any, Tuple
import numpy as np
import yaml
import json
from datetime import datetime
import multiprocessing as mp
from multiprocessing import Event
from queue import Empty
import signal
import sys
from pathlib import Path

class ExperimentWorker:
    def __init__(self, gpu_id: int, job_queue: mp.Queue, result_queue: mp.Queue, stop_event: Event):
        self.gpu_id = gpu_id
        self.job_queue = job_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        
    def run_experiment(self, params: Dict[str, Any], experiment_id: int) -> Tuple[int, Dict[str, Any]]:
        """Run a single experiment with the given parameters."""
        # Construct hydra override arguments
        override_args = [f"{k}={v}" for k, v in params.items()]
        override_args.append(f"gpu={self.gpu_id}")
        
        # Construct the command
        cmd = ["python", "main.py"] + override_args
        
        # Set environment variables
        env = os.environ.copy()
        # env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        
        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True
            )
            
            # Print full output for debugging if there's an error
            if result.returncode != 0:
                print(f"\nExperiment {experiment_id} failed with return code {result.returncode}")
                print("Command:", " ".join(cmd))
                print("\nSTDOUT:")
                print(result.stdout)
                print("\nSTDERR:")
                print(result.stderr)
                raise RuntimeError("Experiment failed")
            
            # Get the last line of output
            output_lines = result.stdout.strip().split('\n')
            if not output_lines:
                raise ValueError("No output from experiment")
            
            last_line = output_lines[-1]
            print(f"\nExperiment {experiment_id} last line output:", last_line)  # Debug print
            
            try:
                test_results = eval(last_line)
                return experiment_id, {"params": params, "results": test_results}
            except SyntaxError:
                print(f"\nFailed to parse results from last line. Full output:")
                print(result.stdout)
                raise
            
        except Exception as e:
            print(f"Error in experiment {experiment_id} on GPU {self.gpu_id}: {str(e)}")
            return experiment_id, {
                "params": params, 
                "results": None, 
                "error": str(e),
                "stdout": result.stdout if 'result' in locals() else None,
                "stderr": result.stderr if 'result' in locals() else None
            }

    def run(self):
        """Main worker loop."""
        while not self.stop_event.is_set():
            try:
                # Get job with timeout to allow for checking stop_event
                experiment_id, params = self.job_queue.get(timeout=1)
                
                # Run the experiment
                result = self.run_experiment(params, experiment_id)
                self.result_queue.put(result)
                
            except Empty:
                continue
            except Exception as e:
                print(f"Worker on GPU {self.gpu_id} encountered error: {str(e)}")
                continue

def worker_process(gpu_id: int, job_queue: mp.Queue, result_queue: mp.Queue, stop_event: Event):
    """Process function for each worker."""
    worker = ExperimentWorker(gpu_id, job_queue, result_queue, stop_event)
    worker.run()

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning script')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1,2,3],
                      help='List of GPU IDs to use')
    parser.add_argument('--experiments-per-gpu', type=int, default=2,
                      help='Number of experiments to run per GPU')
    parser.add_argument('--output-dir', type=str, default='hparam_search_results',
                      help='Directory to save results')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Define hyperparameter search space
    param_grid = {
        'data.task': ['phenotype', 'mortality'],
        'seed': [42, 25, 1],
        'name': ['currentBest'],
        'retrain_teachers': [True],
    }

    # Generate all combinations of hyperparameters
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    total_experiments = len(combinations)

    # Create queues and shared event for job distribution and result collection
    job_queue = mp.Queue()
    result_queue = mp.Queue()
    stop_event = mp.Event()

    # Create worker processes
    workers = []
    for gpu_id in args.gpus:
        for _ in range(args.experiments_per_gpu):
            worker = mp.Process(
                target=worker_process,
                args=(gpu_id, job_queue, result_queue, stop_event)
            )
            workers.append(worker)
            worker.start()

    # Submit all jobs to the queue
    for i, combination in enumerate(combinations):
        params = dict(zip(keys, combination))
        job_queue.put((i, params))

    # Collect results
    results = {}
    completed = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        while completed < total_experiments:
            experiment_id, result = result_queue.get()
            results[experiment_id] = result
            completed += 1
            
            # Save intermediate results
            results_file = output_dir / f"hparam_search_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Print progress
            print(f"Completed {completed}/{total_experiments} experiments")
            
            # If this experiment was successful, print its performance
            if result['results'] is not None:
                print(f"Experiment {experiment_id} PRAUC: {result['results']['overall/PRAUC']:.4f}")

    except KeyboardInterrupt:
        print("\nReceived interrupt, stopping workers...")
        stop_event.set()  # Signal all workers to stop
    
    # Wait for all workers to finish
    for worker in workers:
        worker.join(timeout=5)  # Give workers 5 seconds to finish gracefully
        if worker.is_alive():
            worker.terminate()  # Force terminate if still running

    # Convert results to list and sort by experiment ID
    results_list = [results[i] for i in range(len(results))]
    
    # Find best result
    valid_results = [r for r in results_list if r['results'] is not None]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['results']['overall/PRAUC'])
        print("\nBest hyperparameters:")
        print(json.dumps(best_result, indent=2))
    
    # Save final results
    final_results = {
        'all_results': results_list,
        'best_result': best_result if valid_results else None,
        'completed_experiments': completed,
        'total_experiments': total_experiments
    }
    
    with open(output_dir / f"hparam_search_final_{timestamp}.json", 'w') as f:
        json.dump(final_results, f, indent=2)

if __name__ == '__main__':
    main() 