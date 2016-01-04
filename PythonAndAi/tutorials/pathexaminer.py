import os
from os import listdir
from os.path import isfile, join
import glob

pathfromos="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\libnvvp;C:\ProgramData\Oracle\Java\javapath;C:\Program Files (x86)\Intel\iCLS Client\;C:\Program Files\Intel\iCLS Client\;C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;C:\Program Files\Intel\Intel(R) Management Engine Components\DAL;C:\Program Files\Intel\Intel(R) Management Engine Components\IPT;C:\Program Files (x86)\Intel\Intel(R) Management Engine Components\DAL;C:\Program Files (x86)\Intel\Intel(R) Management Engine Components\IPT;C:\Program Files\TortoiseSVN\bin;C:\Program Files (x86)\Skype\Phone\;C:\mongodb\bin;C:\winbash;C:\winpython;C:\apache-maven-3.3.3\bin;C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common;C:\Anaconda;C:\Anaconda\Scripts;C:\Program Files (x86)\CMake\bin;C:\Program Files (x86)\Windows Kits\8.1\Windows Performance Toolkit\;C:\Program Files\Microsoft SQL Server\110\Tools\Binn\;C:\Program Files (x86)\Microsoft SDKs\TypeScript\1.0\;C:\Program Files\Microsoft SQL Server\120\Tools\Binn\;C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin;C:\Ruby21-x64\bin;C:\Ruby22-x64\bin"

#print(os.path.basename().split(';'))
#print(pathfromos.split(';'))

#for mypath in pathfromos.split(';'):
#	onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
#	print(onlyfiles)

for mypath in pathfromos.split(';'):
	elements = glob.glob(mypath+"\\tar.exe")
	if(len(elements)>0):
		print(elements)