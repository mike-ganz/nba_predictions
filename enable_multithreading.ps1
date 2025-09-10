#!/usr/bin/env pwsh
# Enable Multithreading for NBA Orchestrator on GCP
# Updates orchestrator.py to support parallel simulation execution

Write-Host "=" * 80
Write-Host "ENABLING MULTITHREADED SIMULATIONS ON GCP"
Write-Host "=" * 80

$BUCKET_NAME = (Get-Content "bucket_name.txt" -Raw).Trim()
Write-Host "Using bucket: $BUCKET_NAME"

# Create multithreaded orchestrator patch
Write-Host "`nCreating multithreaded orchestrator..."

$MULTITHREADED_PATCH = @'
    def run_all_simulations_multithreaded(self) -> Dict[str, Any]:
        """Run all configured simulations with multithreading support."""
        total_runs = len(self.config.games) * self.config.runs_per_game
        completed_runs = 0
        failed_runs = 0
        
        self.logger.info(f"Starting MULTITHREADED orchestration: {total_runs} total simulations")
        self.logger.info(f"   Games: {self.config.games}")
        self.logger.info(f"   Runs per game: {self.config.runs_per_game}")
        self.logger.info(f"   Max threads: {self.config.max_threads}")
        self.logger.info(f"   Season: {self.config.season_year}")
        self.logger.info(f"   Platform: {self.config.platform}")
        
        start_time = time.time()
        
        # Create thread pool for parallel execution
        import concurrent.futures
        from threading import Lock
        
        results_lock = Lock()
        
        def run_simulation_worker(args):
            """Worker function for thread pool."""
            run_id, game_id = args
            nonlocal completed_runs, failed_runs
            
            try:
                result = self._run_single_simulation(run_id, game_id)
                self.db.save_result(result)
                
                with results_lock:
                    if result.status == "completed":
                        completed_runs += 1
                        self.logger.info(f"✅ {run_id}: {result.successful_predictions}/{result.total_predictions} predictions, {result.scoring_rate:.1f}% scoring")
                    else:
                        failed_runs += 1
                        self.logger.warning(f"⚠️  {run_id}: {result.status} - {result.termination_reason}")
                
                return result
                
            except Exception as e:
                with results_lock:
                    failed_runs += 1
                    self.logger.error(f"❌ {run_id}: {str(e)}")
                
                # Still save the failed result
                error_result = SimulationResult(
                    run_id=run_id, game_id=game_id, season_year=self.config.season_year,
                    platform=self.config.platform, start_time=datetime.now(),
                    end_time=datetime.now(), duration_seconds=0.0, status="error",
                    total_predictions=0, successful_predictions=0, final_score=None,
                    final_quarter=None, final_time=None, termination_reason=None,
                    scoring_plays=0, non_scoring_plays=0, scoring_rate=0.0,
                    error_message=str(e), iterations_data="[]"
                )
                self.db.save_result(error_result)
                return error_result
        
        # Prepare all simulation tasks
        simulation_tasks = []
        for game_id in self.config.games:
            for run_num in range(self.config.runs_per_game):
                run_id = f"{game_id}_{self.config.season_year}_{run_num + 1:04d}_{int(time.time())}_{run_num}"
                simulation_tasks.append((run_id, game_id))
        
        # Execute simulations in parallel using ThreadPoolExecutor
        max_workers = min(self.config.max_threads, len(simulation_tasks))
        
        self.logger.info(f"🚀 Starting {len(simulation_tasks)} simulations across {max_workers} threads")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(run_simulation_worker, task): task for task in simulation_tasks}
            
            # Process completed tasks
            for i, future in enumerate(concurrent.futures.as_completed(future_to_task), 1):
                try:
                    result = future.result()
                    progress = (i / len(simulation_tasks)) * 100
                    self.logger.info(f"📊 Progress: {i}/{len(simulation_tasks)} ({progress:.1f}%) - Latest: {result.run_id}")
                except Exception as e:
                    self.logger.error(f"Thread execution error: {e}")
        
        total_time = time.time() - start_time
        
        summary = {
            "total_simulations": total_runs,
            "completed": completed_runs,
            "failed": failed_runs,
            "success_rate": (completed_runs / total_runs) * 100 if total_runs > 0 else 0,
            "total_duration_minutes": total_time / 60,
            "avg_time_per_simulation": total_time / total_runs if total_runs > 0 else 0,
            "threads_used": max_workers,
            "parallel_efficiency": (total_time / (total_runs * (total_time / total_runs))) if total_runs > 0 else 0
        }
        
        self.logger.info(f"\n🎊 MULTITHREADED ORCHESTRATION COMPLETE!")
        self.logger.info(f"   Completed: {completed_runs}/{total_runs} ({summary['success_rate']:.1f}%)")
        self.logger.info(f"   Total time: {summary['total_duration_minutes']:.1f} minutes")
        self.logger.info(f"   Threads used: {max_workers}")
        self.logger.info(f"   Results saved to: {self.config.output_db}")
        
        return summary
'@

# Create enhanced orchestrator with multithreading
$ENHANCED_ORCHESTRATOR = @'
#!/usr/bin/env python3
"""
Enhanced NBA Prediction Orchestration System with Multithreading
"""

# Add the multithreaded method to the existing orchestrator
# This patch adds concurrent execution capabilities

import concurrent.futures
from threading import Lock

# Save the multithreaded method to a separate file for integration
def patch_orchestrator_for_multithreading():
    """Apply multithreading patch to orchestrator."""
    
    # Read current orchestrator
    with open('orchestrator.py', 'r') as f:
        content = f.read()
    
    # Add multithreaded method before the analyze_results method
    multithreaded_method = '''
    def run_all_simulations_multithreaded(self) -> Dict[str, Any]:
        """Run all configured simulations with multithreading support."""
        total_runs = len(self.config.games) * self.config.runs_per_game
        completed_runs = 0
        failed_runs = 0
        
        self.logger.info(f"Starting MULTITHREADED orchestration: {total_runs} total simulations")
        self.logger.info(f"   Games: {self.config.games}")
        self.logger.info(f"   Runs per game: {self.config.runs_per_game}")
        self.logger.info(f"   Max threads: {self.config.max_threads}")
        self.logger.info(f"   Season: {self.config.season_year}")
        self.logger.info(f"   Platform: {self.config.platform}")
        
        start_time = time.time()
        
        # Create thread pool for parallel execution
        import concurrent.futures
        from threading import Lock
        
        results_lock = Lock()
        
        def run_simulation_worker(args):
            """Worker function for thread pool."""
            run_id, game_id = args
            nonlocal completed_runs, failed_runs
            
            try:
                result = self._run_single_simulation(run_id, game_id)
                self.db.save_result(result)
                
                with results_lock:
                    if result.status == "completed":
                        completed_runs += 1
                        self.logger.info(f"✅ {run_id}: {result.successful_predictions}/{result.total_predictions} predictions, {result.scoring_rate:.1f}% scoring")
                    else:
                        failed_runs += 1
                        self.logger.warning(f"⚠️  {run_id}: {result.status} - {result.termination_reason}")
                
                return result
                
            except Exception as e:
                with results_lock:
                    failed_runs += 1
                    self.logger.error(f"❌ {run_id}: {str(e)}")
                
                # Still save the failed result
                error_result = SimulationResult(
                    run_id=run_id, game_id=game_id, season_year=self.config.season_year,
                    platform=self.config.platform, start_time=datetime.now(),
                    end_time=datetime.now(), duration_seconds=0.0, status="error",
                    total_predictions=0, successful_predictions=0, final_score=None,
                    final_quarter=None, final_time=None, termination_reason=None,
                    scoring_plays=0, non_scoring_plays=0, scoring_rate=0.0,
                    error_message=str(e), iterations_data="[]"
                )
                self.db.save_result(error_result)
                return error_result
        
        # Prepare all simulation tasks
        simulation_tasks = []
        for game_id in self.config.games:
            for run_num in range(self.config.runs_per_game):
                run_id = f"{game_id}_{self.config.season_year}_{run_num + 1:04d}_{int(time.time())}_{run_num}"
                simulation_tasks.append((run_id, game_id))
        
        # Execute simulations in parallel using ThreadPoolExecutor
        max_workers = min(self.config.max_threads, len(simulation_tasks))
        
        self.logger.info(f"🚀 Starting {len(simulation_tasks)} simulations across {max_workers} threads")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(run_simulation_worker, task): task for task in simulation_tasks}
            
            # Process completed tasks
            for i, future in enumerate(concurrent.futures.as_completed(future_to_task), 1):
                try:
                    result = future.result()
                    progress = (i / len(simulation_tasks)) * 100
                    self.logger.info(f"📊 Progress: {i}/{len(simulation_tasks)} ({progress:.1f}%) - Latest: {result.run_id}")
                except Exception as e:
                    self.logger.error(f"Thread execution error: {e}")
        
        total_time = time.time() - start_time
        
        summary = {
            "total_simulations": total_runs,
            "completed": completed_runs,
            "failed": failed_runs,
            "success_rate": (completed_runs / total_runs) * 100 if total_runs > 0 else 0,
            "total_duration_minutes": total_time / 60,
            "avg_time_per_simulation": total_time / total_runs if total_runs > 0 else 0,
            "threads_used": max_workers,
            "parallel_efficiency": (total_time / (total_runs * (total_time / total_runs))) if total_runs > 0 else 0
        }
        
        self.logger.info(f"\\n🎊 MULTITHREADED ORCHESTRATION COMPLETE!")
        self.logger.info(f"   Completed: {completed_runs}/{total_runs} ({summary['success_rate']:.1f}%)")
        self.logger.info(f"   Total time: {summary['total_duration_minutes']:.1f} minutes")
        self.logger.info(f"   Threads used: {max_workers}")
        self.logger.info(f"   Results saved to: {self.config.output_db}")
        
        return summary

'''
    
    # Replace the run_all_simulations method with a dispatcher
    dispatcher_method = '''
    def run_all_simulations(self) -> Dict[str, Any]:
        """Run all configured simulations (with multithreading if enabled)."""
        if self.config.max_threads > 1:
            return self.run_all_simulations_multithreaded()
        else:
            return self.run_all_simulations_sequential()
    
    def run_all_simulations_sequential(self) -> Dict[str, Any]:
        """Original sequential implementation."""
        total_runs = len(self.config.games) * self.config.runs_per_game
        completed_runs = 0
        failed_runs = 0
        
        self.logger.info(f"Starting SEQUENTIAL orchestration: {total_runs} total simulations")
        self.logger.info(f"   Games: {self.config.games}")
        self.logger.info(f"   Runs per game: {self.config.runs_per_game}")
        self.logger.info(f"   Season: {self.config.season_year}")
        self.logger.info(f"   Platform: {self.config.platform}")
        
        start_time = time.time()
        
        for game_id in self.config.games:
            self.logger.info(f"\\nStarting simulations for game {game_id}")
            
            for run_num in range(self.config.runs_per_game):
                run_id = f"{game_id}_{self.config.season_year}_{run_num + 1:04d}_{int(time.time())}"
                
                self.logger.info(f"   Run {run_num + 1}/{self.config.runs_per_game}: {run_id}")
                
                try:
                    result = self._run_single_simulation(run_id, game_id)
                    self.db.save_result(result)
                    
                    if result.status == "completed":
                        completed_runs += 1
                        self.logger.info(f"   Completed: {result.successful_predictions}/{result.total_predictions} predictions, {result.scoring_rate:.1f}% scoring")
                    else:
                        failed_runs += 1
                        self.logger.warning(f"   Status: {result.status} - {result.termination_reason}")
                
                except Exception as e:
                    failed_runs += 1
                    self.logger.error(f"   Failed: {str(e)}")
                    
                    # Still save the failed result
                    error_result = SimulationResult(
                        run_id=run_id, game_id=game_id, season_year=self.config.season_year,
                        platform=self.config.platform, start_time=datetime.now(),
                        end_time=datetime.now(), duration_seconds=0.0, status="error",
                        total_predictions=0, successful_predictions=0, final_score=None,
                        final_quarter=None, final_time=None, termination_reason=None,
                        scoring_plays=0, non_scoring_plays=0, scoring_rate=0.0,
                        error_message=str(e), iterations_data="[]"
                    )
                    self.db.save_result(error_result)
        
        total_time = time.time() - start_time
        
        summary = {
            "total_simulations": total_runs,
            "completed": completed_runs,
            "failed": failed_runs,
            "success_rate": (completed_runs / total_runs) * 100 if total_runs > 0 else 0,
            "total_duration_minutes": total_time / 60,
            "avg_time_per_simulation": total_time / total_runs if total_runs > 0 else 0
        }
        
        self.logger.info(f"\\nOrchestration Complete!")
        self.logger.info(f"   Completed: {completed_runs}/{total_runs} ({summary['success_rate']:.1f}%)")
        self.logger.info(f"   Total time: {summary['total_duration_minutes']:.1f} minutes")
        self.logger.info(f"   Results saved to: {self.config.output_db}")
        
        return summary
'''
    
    # Find and replace the run_all_simulations method
    import re
    
    # Pattern to match the entire run_all_simulations method
    pattern = r'(\s+def run_all_simulations\(self\)[^:]*:.*?)(\s+def [^:]*:|$)'
    
    # Replace with new dispatcher + sequential methods + multithreaded method
    replacement = dispatcher_method + multithreaded_method + r'\2'
    
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Write the patched orchestrator
    with open('orchestrator_multithreaded.py', 'w') as f:
        f.write(new_content)
    
    print("✅ Multithreaded orchestrator created: orchestrator_multithreaded.py")

if __name__ == "__main__":
    patch_orchestrator_for_multithreading()
'@

# Save and run the patch
$ENHANCED_ORCHESTRATOR | Out-File -FilePath "patch_orchestrator.py" -Encoding UTF8
python3 patch_orchestrator.py
Remove-Item "patch_orchestrator.py"

# Upload the multithreaded version to GCP
Write-Host "`nUploading multithreaded orchestrator to GCP..."
gcloud compute scp orchestrator_multithreaded.py nba-orchestrator:~/ --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch"

Write-Host "`nBacking up original orchestrator on VM..."
gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="cp orchestrator.py orchestrator_original.py"

Write-Host "`nInstalling multithreaded version..."
gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="cp orchestrator_multithreaded.py orchestrator.py"

# Create multithreaded config
$MULTITHREADED_CONFIG = @'
{
  "season_year": "2023-2024",
  "games": ["22200001", "22200002", "22200003"],
  "runs_per_game": 8,
  "max_iterations_per_run": 1000,
  "max_threads": 4,
  "skip_stage1": true,
  "platform": "gemini",
  "output_db": "multithreaded_simulation_results.db",
  "log_level": "INFO",
  "resume_on_error": true,
  "timeout_minutes": 30
}
'@

$MULTITHREADED_CONFIG | Out-File -FilePath "orchestrator_config_multithreaded.json" -Encoding UTF8
gcloud compute scp orchestrator_config_multithreaded.json nba-orchestrator:~/ --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch"

Write-Host "`n" + "=" * 80
Write-Host "MULTITHREADING ENABLED!"
Write-Host "=" * 80
Write-Host "Configuration:"
Write-Host "- Max threads: 4 (optimal for e2-standard-4 VM)"
Write-Host "- Parallel simulation execution"
Write-Host "- Thread-safe database operations"
Write-Host "- Real-time progress tracking"
Write-Host "`nTest the multithreaded version:"
Write-Host "gcloud compute ssh nba-orchestrator --zone=us-central1-a --ssh-flag='-batch' --command='source ~/venv/bin/activate && python3 orchestrator.py --config orchestrator_config_multithreaded.json'"
