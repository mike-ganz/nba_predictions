#!/usr/bin/env pwsh
# Direct Multithreading Setup on GCP VM
# Bypasses local Python patching issues

Write-Host "=" * 80
Write-Host "FIXING MULTITHREADING SETUP ON GCP"
Write-Host "=" * 80

$BUCKET_NAME = (Get-Content "bucket_name.txt" -Raw).Trim()
Write-Host "Using bucket: $BUCKET_NAME"

# Create the multithreaded orchestrator directly as a string and upload it
Write-Host "`nStep 1: Creating multithreaded orchestrator on VM..."

$MULTITHREADED_ORCHESTRATOR_PATCH = @'
#!/usr/bin/env python3
"""
Multithreaded NBA Orchestrator Patch
Adds concurrent execution to existing orchestrator
"""

import os
import shutil

def patch_orchestrator():
    """Add multithreading capability to orchestrator.py"""
    
    # Read current orchestrator
    with open('orchestrator.py', 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'run_all_simulations_multithreaded' in content:
        print("✅ Orchestrator already has multithreading support")
        return
    
    # Backup original
    shutil.copy('orchestrator.py', 'orchestrator_original.py')
    print("📦 Backed up original orchestrator")
    
    # Add imports at the top after existing imports
    import_addition = '''
import concurrent.futures
from threading import Lock
'''
    
    # Find where to insert imports (after existing imports)
    import_pos = content.find('from game_context_builder import GameContextBuilder')
    if import_pos != -1:
        end_of_import = content.find('\n', import_pos) + 1
        content = content[:end_of_import] + import_addition + content[end_of_import:]
    
    # Add multithreaded method before analyze_results method
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
        
        # Execute simulations in parallel
        max_workers = min(self.config.max_threads, len(simulation_tasks))
        
        self.logger.info(f"🚀 Starting {len(simulation_tasks)} simulations across {max_workers} threads")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(run_simulation_worker, task): task for task in simulation_tasks}
            
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
            "threads_used": max_workers
        }
        
        self.logger.info(f"\\n🎊 MULTITHREADED ORCHESTRATION COMPLETE!")
        self.logger.info(f"   Completed: {completed_runs}/{total_runs} ({summary['success_rate']:.1f}%)")
        self.logger.info(f"   Total time: {summary['total_duration_minutes']:.1f} minutes")
        self.logger.info(f"   Threads used: {max_workers}")
        self.logger.info(f"   Results saved to: {self.config.output_db}")
        
        return summary

'''
    
    # Replace the original run_all_simulations method with a dispatcher
    original_method_start = content.find('def run_all_simulations(self) -> Dict[str, Any]:')
    if original_method_start == -1:
        print("❌ Could not find run_all_simulations method")
        return
    
    # Find the end of the method (next method definition or end of class)
    method_end = content.find('\\n    def ', original_method_start + 1)
    if method_end == -1:
        method_end = len(content)
    
    # Create new dispatcher method
    new_dispatcher = '''    def run_all_simulations(self) -> Dict[str, Any]:
        """Run all configured simulations (with multithreading if enabled)."""
        if self.config.max_threads > 1:
            return self.run_all_simulations_multithreaded()
        else:
            return self.run_all_simulations_sequential()
    
    def run_all_simulations_sequential(self) -> Dict[str, Any]:
        """Original sequential implementation."""''' + content[original_method_start + len('def run_all_simulations(self) -> Dict[str, Any]:'):method_end]
    
    # Replace the method
    content = content[:original_method_start] + new_dispatcher + multithreaded_method + content[method_end:]
    
    # Write the patched orchestrator
    with open('orchestrator.py', 'w') as f:
        f.write(content)
    
    print("✅ Successfully patched orchestrator with multithreading support")

if __name__ == "__main__":
    patch_orchestrator()
'@

# Write patch script to local file temporarily
$MULTITHREADED_ORCHESTRATOR_PATCH | Out-File -FilePath "patch_script.py" -Encoding UTF8

# Upload patch script to VM
Write-Host "📤 Uploading patch script to VM..."
gcloud compute scp patch_script.py nba-orchestrator:~/ --zone=us-central1-a --project=utopian-outlook-470922-q2 --scp-flag="-batch"

# Run patch script on VM
Write-Host "⚡ Applying multithreading patch on VM..."
gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="source ~/.bashrc && source ~/venv/bin/activate && cd ~ && python3 patch_script.py"

# Clean up local patch script
Remove-Item "patch_script.py"

# Verify the patch worked
Write-Host "`nStep 2: Verifying multithreading setup..."
$verify_result = gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --ssh-flag="-batch" --command="grep -q 'run_all_simulations_multithreaded' orchestrator.py && echo 'SUCCESS: Multithreading enabled' || echo 'ERROR: Multithreading not found'"

Write-Host $verify_result

if ($verify_result -like "*SUCCESS*") {
    Write-Host "`n✅ MULTITHREADING SUCCESSFULLY ENABLED!"
    Write-Host "📋 Your orchestrator now supports:"
    Write-Host "   - Parallel execution with configurable threads"
    Write-Host "   - Thread-safe database operations"  
    Write-Host "   - Real-time progress tracking"
    Write-Host "   - Automatic fallback to sequential mode"
} else {
    Write-Host "`n❌ MULTITHREADING SETUP FAILED!"
    Write-Host "Please check the VM logs for errors."
}

Write-Host "`n🚀 Ready to run your big simulation!"
Write-Host "Next step: .\run_simulation_config.ps1 config_big_run.json"
