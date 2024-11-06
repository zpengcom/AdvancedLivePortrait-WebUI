if (-Not (Test-Path -Path "$PSScriptRoot\venv\Scripts")) {
    Write-Output "Creating venv..."
    & .\install.ps1
}

& .\venv\Scripts\Activate.ps1

python .\app.py $args

Write-Host "Launching the app"
Pause
