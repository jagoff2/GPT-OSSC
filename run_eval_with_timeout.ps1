param(
    [int]$TimeoutSeconds = 900,
    [string]$LogPath = "G:\OSSC\eval_latest_output.log"
)

$ErrorActionPreference = 'Stop'

$cwd = "G:\OSSC"
$stdoutPath = [System.IO.Path]::ChangeExtension($LogPath, ".stdout.log")
$stderrPath = [System.IO.Path]::ChangeExtension($LogPath, ".stderr.log")

foreach ($path in @($LogPath, $stdoutPath, $stderrPath)) {
    if (Test-Path $path) {
        Remove-Item $path -Force
    }
}

$process = Start-Process -FilePath "py" -ArgumentList "-m cli.main eval" -WorkingDirectory $cwd -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru -NoNewWindow

try {
    try {
        Wait-Process -InputObject $process -Timeout $TimeoutSeconds -ErrorAction Stop
    } catch [System.Management.Automation.TimeoutException] {
        try {
            Stop-Process -InputObject $process -Force -ErrorAction SilentlyContinue
        } catch {
            Write-Warning "Failed to terminate process after timeout: $_"
        }
        throw "Command timed out after $TimeoutSeconds seconds."
    }

    $exitCode = $process.ExitCode
}
finally {
    $process.Dispose()
}

$combined = @()
if (Test-Path $stdoutPath) {
    $combined += Get-Content $stdoutPath
}
if (Test-Path $stderrPath) {
    $combined += (Get-Content $stderrPath | ForEach-Object { "[stderr] $_" })
}

$combined | Set-Content $LogPath

Write-Host "===== Command Output ====="
Get-Content $LogPath | ForEach-Object { Write-Host $_ }
Write-Host "===== End Output ====="
Write-Host "ExitCode: $exitCode"

exit $exitCode
