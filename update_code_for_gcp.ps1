# Update Code for GCP - Modify config/settings.py for Cloud Storage
# This must be run BEFORE creating the deployment package

Write-Host "Updating code for GCP compatibility..." -ForegroundColor Green

# Backup original config file
if (Test-Path "config\settings.py") {
    Copy-Item "config\settings.py" "config\settings.py.backup"
    Write-Host "SUCCESS: Backed up original config\settings.py" -ForegroundColor Green
} else {
    Write-Host "ERROR: config\settings.py not found!" -ForegroundColor Red
    exit 1
}

# Read current config file
$configContent = Get-Content "config\settings.py" -Raw

# Create the new _get_file_mapping method
$newFileMapping = @'
    def _get_file_mapping(self) -> Dict[str, str]:
        """Get mapping of season years to file paths."""
        
        # Check if running on GCP (environment variable set)
        bucket_name = os.getenv('NBA_DATA_BUCKET')
        
        if bucket_name:
            # Running on GCP - return Cloud Storage paths
            base_path = f"gs://{bucket_name}/data/play_by_play/historical"
            return {
                "2022-2023": f"{base_path}/[10-18-2022]-[06-12-2023]-combined-stats.csv",
                "2023-2024": f"{base_path}/[10-24-2023]-[06-17-2024]-combined-stats.csv",
                "2024-2025": f"{base_path}/[10-22-2024]-[06-22-2025]-combined-stats.csv"
            }
        else:
            # Running locally - return local paths (unchanged)
            return {
                "2022-2023": os.path.join(
                    self.play_by_play_dir, 
                    "[10-18-2022]-[06-12-2023]-combined-stats.csv"
                ),
                "2023-2024": os.path.join(
                    self.play_by_play_dir,
                    "[10-24-2023]-[06-17-2024]-combined-stats.csv"
                ),
                "2024-2025": os.path.join(
                    self.play_by_play_dir,
                    "[10-22-2024]-[06-22-2025]-combined-stats.csv"
                )
            }
'@

# Create the new get_play_by_play_file_path method
$newGetMethod = @'
    def get_play_by_play_file_path(self, season_year: Optional[str] = None) -> str:
        """
        Get the file path for play-by-play data for a specific season.
        
        Args:
            season_year: Season year in format "YYYY-YYYY". If None, uses current season.
            
        Returns:
            str: Full file path to the play-by-play data
            
        Raises:
            ValueError: If season year is not supported
            FileNotFoundError: If the data file doesn't exist (local files only)
        """
        if season_year is None:
            season_year = self._season_year
            
        file_mapping = self._get_file_mapping()
        
        if season_year not in file_mapping:
            raise ValueError(
                f"Season year {season_year} not supported. "
                f"Available options: {list(file_mapping.keys())}"
            )
        
        file_path = file_mapping[season_year]
        
        # Only check existence for local files, not Cloud Storage
        if not file_path.startswith('gs://') and not os.path.exists(file_path):
            raise FileNotFoundError(f"Play-by-play data file not found: {file_path}")
            
        return file_path
'@

# Replace the old methods with new ones
Write-Host "Updating _get_file_mapping method..." -ForegroundColor Cyan

# Find and replace the _get_file_mapping method
$pattern = 'def _get_file_mapping\(self\)[\s\S]*?(?=def |\Z)'
if ($configContent -match $pattern) {
    $configContent = $configContent -replace $pattern, ($newFileMapping + "`n`n    ")
    Write-Host "SUCCESS: Updated _get_file_mapping method" -ForegroundColor Green
} else {
    Write-Host "WARNING: Could not find _get_file_mapping method pattern" -ForegroundColor Yellow
}

Write-Host "Updating get_play_by_play_file_path method..." -ForegroundColor Cyan

# Find and replace the get_play_by_play_file_path method
$pattern2 = 'def get_play_by_play_file_path\(self[\s\S]*?(?=def |\Z)'
if ($configContent -match $pattern2) {
    $configContent = $configContent -replace $pattern2, ($newGetMethod + "`n`n    ")
    Write-Host "SUCCESS: Updated get_play_by_play_file_path method" -ForegroundColor Green
} else {
    Write-Host "WARNING: Could not find get_play_by_play_file_path method pattern" -ForegroundColor Yellow
}

# Write updated content back to file
$configContent | Out-File -FilePath "config\settings.py" -Encoding UTF8

Write-Host "`nSUCCESS: Code updated for GCP!" -ForegroundColor Green
Write-Host "Changes made:" -ForegroundColor Yellow
Write-Host "  - Modified _get_file_mapping to support Cloud Storage paths" -ForegroundColor White
Write-Host "  - Updated get_play_by_play_file_path to handle gs:// URLs" -ForegroundColor White
Write-Host "  - Backup saved as config\settings.py.backup" -ForegroundColor White
Write-Host "`nReady to create deployment package!" -ForegroundColor Green

# Explicitly exit with success code
exit 0
