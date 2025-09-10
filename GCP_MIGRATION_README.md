# NBA Predictions GCP Migration - Complete Guide

## 🎯 Overview
This package contains all the PowerShell scripts needed to migrate your NBA prediction system to Google Cloud Platform. Everything is pre-configured with your specific settings:

- **Project**: utopian-outlook-470922-q2
- **Account**: ganzy9@gmail.com
- **Gemini Models**: 4062830704562536448, 8359264749073989632
- **VM Zone**: us-central1-a
- **Naming Convention**: nba_predictions

## 📁 Script Files

### 🚀 Main Execution Scripts

| Script | Purpose | Runtime |
|--------|---------|---------|
| `run_full_setup.ps1` | **🎯 Run this first!** Complete automated setup | 15-20 min |
| `start_simulation.ps1` | Start NBA prediction simulations | 2 min |
| `monitor.ps1` | Interactive monitoring dashboard | Continuous |

### 🔧 Individual Setup Scripts (run automatically by main script)

| Script | Purpose | What it does |
|--------|---------|-------------|
| `update_code_for_gcp.ps1` | Code modifications | Updates `config/settings.py` for Cloud Storage |
| `setup_phase1.ps1` | Data & Storage | Creates bucket, uploads data, sets up permissions |
| `setup_phase2.ps1` | VM Setup | Creates VM, packages code |
| `deploy.ps1` | Deployment | Deploys code to VM, configures environment |
| `test_system.ps1` | System Testing | Validates all components work |

## 🚀 Quick Start (Just 3 Steps!)

### Step 1: Run Complete Setup
```powershell
.\run_full_setup.ps1
```
**This does everything automatically!** ☕ Grab coffee, takes 15-20 minutes.

### Step 2: Start Simulations
```powershell
.\start_simulation.ps1
```
Choose your simulation type and it launches in the cloud.

### Step 3: Monitor Progress
```powershell
.\monitor.ps1
```
Real-time dashboard showing status, logs, database stats.

## 📋 What Gets Created

### ☁️ Google Cloud Resources
- **Cloud Storage Bucket**: `nba-predictions-[timestamp]` (name saved to `bucket_name.txt`)
- **Compute VM**: `nba-orchestrator` (e2-standard-4, us-central1-a)
- **Service Account**: `nba-orchestrator@utopian-outlook-470922-q2.iam.gserviceaccount.com`
- **IAM Permissions**: Storage access + AI Platform access

### 📁 Local Files Created
- `bucket_name.txt` - Your bucket name (needed by other scripts)
- `requirements.txt` - Python dependencies
- `nba_predictions_gcp.zip` - Deployment package
- `config/settings.py.backup` - Backup of original config

## 🎮 Simulation Options

When you run `start_simulation.ps1`, you can choose:

1. **Quick Test** - 3 games, 2 runs each, 100 iterations (~10 minutes)
2. **Medium Run** - 5 games, 5 runs each, 500 iterations (~30 minutes)  
3. **Full Simulation** - Uses your `orchestrator_config.json` (hours)
4. **Custom** - Specify your own parameters

## 🔍 Monitoring Features

The `monitor.ps1` dashboard shows:
- ✅ VM status
- 🔄 Running processes  
- 📋 Recent log entries
- 🗄️ Database statistics (total runs, completed, recent)
- ☁️ Cloud Storage usage
- 🎮 Interactive options (view logs, SSH, download database)

## 💰 Cost Breakdown

**Monthly Estimates**:
- VM (e2-standard-4): ~$55/month
- Cloud Storage: ~$8-15/month  
- Data transfer: ~$2-5/month
- **Total**: ~$65-75/month

**Cost Controls**:
- VM only runs when you need it
- Stop VM when not in use: `gcloud compute instances stop nba-orchestrator --zone=us-central1-a`
- Start when needed: `gcloud compute instances start nba-orchestrator --zone=us-central1-a`

## 🛠️ Manual Commands (if needed)

### VM Management
```powershell
# Start VM
gcloud compute instances start nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2

# Stop VM (saves money!)
gcloud compute instances stop nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2

# SSH to VM
gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2
```

### Data Management
```powershell
# Download database
gcloud compute scp nba-orchestrator:~/enhanced_simulation_results.db ./results_backup.db --zone=us-central1-a --project=utopian-outlook-470922-q2

# View logs
gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --command="tail -f orchestrator_gcp.log"
```

## ❌ Troubleshooting

### Common Issues

**"gcloud not found"**
- Install Google Cloud CLI from: https://cloud.google.com/sdk/docs/install-windows

**"Authentication failed"**  
```powershell
gcloud auth login
gcloud config set project utopian-outlook-470922-q2
```

**"VM won't start"**
- Check quotas in GCP Console
- Try different zone: `--zone=us-central1-b`

**"Data loading failed"**
- Check bucket name in `bucket_name.txt`
- Verify files uploaded: `gsutil ls gs://[bucket-name]/data/`

**"Simulation won't start"**
- Run `.\test_system.ps1` to diagnose
- Check Gemini endpoint configuration

### Getting Help

**Check system status**:
```powershell
.\test_system.ps1
```

**View detailed logs**:
```powershell
gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --command="tail -50 orchestrator_gcp.log"
```

**Check environment**:
```powershell
gcloud compute ssh nba-orchestrator --zone=us-central1-a --project=utopian-outlook-470922-q2 --command="source ~/.bashrc && env | grep NBA"
```

## 🎯 Success Indicators

You know it's working when:
- ✅ `run_full_setup.ps1` completes without errors
- ✅ `test_system.ps1` shows all green checkmarks
- ✅ `monitor.ps1` shows running processes
- ✅ Database row count increases over time
- ✅ Log files show "prediction completed" messages

## 🔄 Next Steps After Setup

1. **Run your first simulation**: `.\start_simulation.ps1`
2. **Set up monitoring**: Keep `.\monitor.ps1` running in another window
3. **Schedule regular backups**: Database gets backed up automatically to Cloud Storage
4. **Optimize costs**: Stop VM when not needed
5. **Scale up**: Can upgrade VM size for faster simulations

---

## 📞 Support

Your system is configured for:
- **Project**: utopian-outlook-470922-q2
- **Gemini Endpoints**: 4062830704562536448, 8359264749073989632
- **Zone**: us-central1-a (same as your Gemini models)

All scripts are pre-configured - just run `.\run_full_setup.ps1` and you're ready to go! 🚀
