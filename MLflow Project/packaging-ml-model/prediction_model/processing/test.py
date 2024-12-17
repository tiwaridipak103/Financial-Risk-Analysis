import os
from pathlib import Path

print('dipak')
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
#sys.path.append(str(PACKAGE_ROOT))
print(PACKAGE_ROOT)