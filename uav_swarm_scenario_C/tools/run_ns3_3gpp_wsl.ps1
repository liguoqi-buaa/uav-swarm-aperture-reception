$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repoRootForWsl = $repoRoot -replace '\\', '/'
$repoWsl = (wsl wslpath -a "$repoRootForWsl").Trim()
$ns3Root = '$HOME/ns-allinone-3.38/ns-3.38'

$script = @"
set -euo pipefail
if [ ! -x $ns3Root/ns3 ]; then
  echo "NS-3.38 is not built at $ns3Root. Run tools/bootstrap_ns3_wsl.ps1 first." >&2
  exit 2
fi
cp "$repoWsl/ns3/scratch/tle-uav-fanet-3gpp.cc" "$ns3Root/scratch/tle-uav-fanet-3gpp.cc"
cd "$ns3Root"
./ns3 run scratch/tle-uav-fanet-3gpp | tee "$repoWsl/reports/ns3_3gpp_report.txt"
"@

$bytes = [System.Text.Encoding]::UTF8.GetBytes($script)
$base64 = [Convert]::ToBase64String($bytes)
$launcher = "printf '%s' '$base64' | base64 -d > /tmp/run_ns3_3gpp_wsl.sh && chmod +x /tmp/run_ns3_3gpp_wsl.sh && /tmp/run_ns3_3gpp_wsl.sh"
wsl bash -lc $launcher
