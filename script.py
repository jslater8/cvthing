import os
import subprocess
from colorama import Fore, Back, Style


for files in os.listdir('dartImages'):
	print(Fore.GREEN + files + Style.RESET_ALL)
	subprocess.call(['./dart','dartImages/' + files,'dartDetect/' + files[:-4] + 'Output.jpg', 'dartDetect/' + files])

