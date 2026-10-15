param(
    [string]$Ns3Version = "3.38",
    [string]$InstallRoot = '$HOME'
)

$ErrorActionPreference = "Stop"

Write-Host "Installing NS-3 $Ns3Version inside WSL Ubuntu."
Write-Host "This command requires your WSL sudo password for apt packages."

$script = @"
set -euo pipefail
sudo apt-get update
sudo apt-get install -y git g++ cmake ninja-build python3 python3-dev python3-pip pkg-config sqlite3 libsqlite3-dev libxml2 libxml2-dev
cd $InstallRoot
if [ ! -d ns-allinone-$Ns3Version ]; then
  wget -O ns-allinone-$Ns3Version.tar.bz2 https://www.nsnam.org/releases/ns-allinone-$Ns3Version.tar.bz2
  tar xjf ns-allinone-$Ns3Version.tar.bz2
fi
cd ns-allinone-$Ns3Version/ns-$Ns3Version
./ns3 configure --enable-examples --enable-tests
./ns3 build
./ns3 run hello-simulator
"@

wsl bash -lc $script

