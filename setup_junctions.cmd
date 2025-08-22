@echo off
setlocal
set ROOT=%~dp0
mkdir "C:\Users\muhenghe\Documents\BYLW" 2>nul
rmdir "C:\Users\muhenghe\Documents\BYLW\start" 2>nul
rmdir "C:\Users\muhenghe\Documents\BYLW\项目初始" 2>nul
mklink /J "C:\Users\muhenghe\Documents\BYLW\start" "%ROOT%start"
mklink /J "C:\Users\muhenghe\Documents\BYLW\项目初始" "%ROOT%项目初始"
echo ✅ Junctions ready.
