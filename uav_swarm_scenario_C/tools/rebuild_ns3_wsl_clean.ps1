$ErrorActionPreference = "Stop"

$bashScript = Join-Path $env:TEMP "rebuild_ns3_wsl_clean.sh"

@'
#!/usr/bin/env bash
set -euo pipefail

echo "[1/5] Installing native WSL build dependencies..."
sudo apt-get install -y \
  git g++ cmake ninja-build python3 python3-dev python3-pip pkg-config \
  sqlite3 libsqlite3-dev libxml2 libxml2-dev libboost-all-dev wget bzip2

NS3_ROOT="$HOME/ns-allinone-3.38/ns-3.38"
if [ ! -d "$NS3_ROOT" ]; then
  echo "NS-3.38 source tree is missing at $NS3_ROOT" >&2
  echo "Run tools/bootstrap_ns3_wsl.ps1 first, or download ns-allinone-3.38 manually." >&2
  exit 2
fi

case "$NS3_ROOT" in
  "$HOME"/ns-allinone-3.38/ns-3.38) ;;
  *) echo "Refusing to clean unexpected path: $NS3_ROOT" >&2; exit 3 ;;
esac

echo "[2/5] Cleaning stale CMake cache that referenced Windows Anaconda headers..."
cd "$NS3_ROOT"
rm -rf cmake-cache build

echo "[2.5/5] Applying GCC 13 compatibility patch for ns-3.38 WiFi headers..."
if ! grep -q '#include <cstdint>' src/wifi/model/sta-wifi-mac.h; then
  sed -i '/#include <variant>/a #include <cstdint>' src/wifi/model/sta-wifi-mac.h
fi

clean_env() {
  env \
    -u CPATH \
    -u C_INCLUDE_PATH \
    -u CPLUS_INCLUDE_PATH \
    -u LIBRARY_PATH \
    -u CMAKE_PREFIX_PATH \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    "$@"
}

echo "[3/5] Configuring NS-3.38 with WSL-native dependencies only..."
clean_env ./ns3 configure --enable-examples --enable-tests -- \
  -DCMAKE_IGNORE_PREFIX_PATH="/mnt/d/Anaconda;/mnt/c;/mnt/d"

echo "[4/5] Building NS-3.38..."
clean_env ./ns3 build

echo "[5/5] Running hello-simulator..."
clean_env ./ns3 run hello-simulator

echo "NS-3.38 clean rebuild completed."
'@ | Set-Content -LiteralPath $bashScript -Encoding UTF8

$scriptContent = Get-Content -LiteralPath $bashScript -Raw
$bytes = [System.Text.Encoding]::UTF8.GetBytes($scriptContent)
$base64 = [Convert]::ToBase64String($bytes)
$launcher = "printf '%s' '$base64' | base64 -d > /tmp/rebuild_ns3_wsl_clean.sh && chmod +x /tmp/rebuild_ns3_wsl_clean.sh && /tmp/rebuild_ns3_wsl_clean.sh"
wsl bash -lc $launcher
