import json
notebook_path = 'Exp8/Exp8.ipynb'
with open(notebook_path, 'r') as f:
    nb = json.load(f)
new_source = [
    "# Load dataset\n",
    "import pandas as pd\n",
    "base_dir = 'dataset/dataset'\n",
    "X_train = pd.read_csv(f'{base_dir}/train/X_train.txt', sep=r'\\s+', header=None)\n",
    "y_train = pd.read_csv(f'{base_dir}/train/y_train.txt', sep=r'\\s+', header=None)\n",
    "X_test = pd.read_csv(f'{base_dir}/test/X_test.txt', sep=r'\\s+', header=None)\n",
    "y_test = pd.read_csv(f'{base_dir}/test/y_test.txt', sep=r'\\s+', header=None)\n",
    "\n",
    "X = pd.concat([X_train, X_test], ignore_index=True)\n",
    "y_true = pd.concat([y_train, y_test], ignore_index=True)[0]\n",
    "\n",
    "# Map activity labels\n",
    "activity_labels = pd.read_csv(f'{base_dir}/activity_labels.txt', sep=r'\\s+', header=None, index_col=0)[1].to_dict()\n",
    "df = X.copy()\n",
    "df['Activity'] = y_true.map(activity_labels)\n",
    "\n",
    "# Encode categorical labels\n",
    "le = LabelEncoder()\n",
    "y_encoded = le.fit_transform(df['Activity'])\n",
    "y_true = df['Activity']\n",
    "\n",
    "# Apply normalization/standardization\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n",
    "\n",
    "print(f'Dataset shape: {X_scaled.shape}')\n"
]
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and ('train_df = pd.read_csv' in ''.join(cell['source']) or 'Generating synthetic data' in ''.join(cell['source'])):
        cell['source'] = new_source
        break
with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)