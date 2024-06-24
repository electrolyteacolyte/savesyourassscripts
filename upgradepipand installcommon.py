import subprocess

# Upgrade pip to the latest version
subprocess.run(['python', '-m', 'pip', 'install', '--upgrade', 'pip'])

# List of commonly used libraries
libraries = [
    'numpy',
    'pandas',
    'matplotlib',
    'scipy',
    'scikit-learn',
    'tensorflow',
    'torch',
    'keras',
    'nltk',
    'requests',
    'sqlalchemy',
    'WMI',
    'beautifulsoup4',
    'seaborn',
    'sqlalchemy',
    'opencv-python',
    'pillow',
    'django',
    'flask',
    'pyyaml',
    'pytest',
    'coverage',
    'sphinx',
    'jupyterlab',
    'plotly',
    'bokeh',
    'dash',
    'tweepy',
    'openpyxl',
    'xlrd',
    'xgboost',
    'catboost',
    'pymongo',
    'redis',
    'networkx',
    'pydot',
    'psycopg2',
    'boto3',
    'flask-restful',
    'flask-jwt-extended',
    'flask-cors',
    'marshmallow',
    'pyjwt',
    'google-cloud-storage',
    'google-cloud-bigquery',
    'google-cloud-pubsub',
    'google-cloud-translate',
    'google-cloud-vision',
    'google-cloud-language',
    'google-cloud-speech',
    'pyttsx3',
    'gTTS',
    'moviepy',
    'pytube',
    'pydub',
    'pysocks',
    'asyncio',
    'websockets',
    'paramiko',
    'pycaw',
    'folium',
    'shapely',
    'tqdm',
    'pandasql',
    'python-Levenshtein',
    'i2plib',
    'fcp',
    'phonenumbers',
    'pywin32',
    'stem',
    'uuid',
    'bleak',
    'scapy',
    'plyer',
    'pywebview',
    'dnspython',
    'thread6',
    'speechrecognition',
    'pygame',
    'pyinstaller'
]

# Create a virtual environment
subprocess.run(['python', '-m', 'venv', 'myenv'])

# Activate the virtual environment
subprocess.run(['myenv\\Scripts\\activate.bat'], shell=True)

# Install the libraries within the virtual environment
for lib in libraries:
    subprocess.run(['pip', 'install', lib])

# Deactivate the virtual environment
subprocess.run(['deactivate'])

print("Installation complete.")
