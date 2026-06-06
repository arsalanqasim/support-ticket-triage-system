import json
import os

def update_tfidf_notebook(path):
    print(f"Modifying TF-IDF notebook: {path}...")
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return
        
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    updated_imports = False
    updated_training = False
    
    # Find and update cells
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # 1. Update imports cell
            if "from sklearn.model_selection import train_test_split" in source and "import mlflow" not in source:
                cell['source'] = [
                    "import pandas as pd\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.linear_model import LogisticRegression\n",
                    "from sklearn.svm import LinearSVC\n",
                    "from sklearn.naive_bayes import MultinomialNB\n",
                    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
                    "from sklearn.pipeline import Pipeline\n",
                    "from sklearn.metrics import accuracy_score, f1_score\n",
                    "from sklearn.preprocessing import MultiLabelBinarizer\n",
                    "from sklearn.multiclass import OneVsRestClassifier\n",
                    "import mlflow\n",
                    "import mlflow.sklearn\n"
                ]
                updated_imports = True
                print("-> Updated imports cell.")
            
            # 2. Update model training loop cell
            elif "for name, clf in models.items():" in source and "mlflow.start_run" not in source:
                cell['source'] = [
                    "# Set the MLflow experiment name\n",
                    "mlflow.set_experiment(\"tfidf_baseline\")\n",
                    "\n",
                    "X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size = 0.2, random_state = 42)\n",
                    "\n",
                    "models = {\n",
                    "    \"logreg\": OneVsRestClassifier(LogisticRegression(max_iter=3000, class_weight=\"balanced\")),\n",
                    "    \"svm\": OneVsRestClassifier(LinearSVC(class_weight=\"balanced\"))\n",
                    "}\n",
                    "\n",
                    "results = []\n",
                    "\n",
                    "for name, clf in models.items():\n",
                    "    # Start an MLflow run for each model comparison\n",
                    "    with mlflow.start_run(run_name=name):\n",
                    "        # Log model type\n",
                    "        mlflow.log_param(\"model_type\", name)\n",
                    "        \n",
                    "        pipe = Pipeline([\n",
                    "            (\"tdif\", TfidfVectorizer(\n",
                    "                lowercase=True,\n",
                    "                stop_words=\"english\",\n",
                    "                ngram_range=(1,2),\n",
                    "                min_df=2, \n",
                    "                max_features=50000,\n",
                    "                sublinear_tf=True\n",
                    "            )),\n",
                    "            (\"clf\", clf)\n",
                    "        ])\n",
                    "\n",
                    "        pipe.fit(X_train, y_train)\n",
                    "        y_pred = pipe.predict(X_test)\n",
                    "\n",
                    "        acc_score = accuracy_score(y_test, y_pred)\n",
                    "        f1_val = f1_score(y_test, y_pred, average='macro')\n",
                    "\n",
                    "        # Log metrics to MLflow\n",
                    "        mlflow.log_metric(\"accuracy\", acc_score)\n",
                    "        mlflow.log_metric(\"macro_f1\", f1_val)\n",
                    "        \n",
                    "        # Log the trained pipeline model\n",
                    "        mlflow.sklearn.log_model(pipe, artifact_path=\"model\")\n",
                    "\n",
                    "        results.append((name, acc_score, f1_val))\n",
                    "        print(f\"\\n{name}\")\n",
                    "        print(\"accuracy:\", acc_score)\n",
                    "        print(\"macro f1_val:\", f1_val)\n",
                    "        from sklearn.metrics import classification_report\n",
                    "        print(classification_report(y_test, y_pred, target_names=mlb.classes_))\n"
                ]
                updated_training = True
                print("-> Updated model training loop cell.")

    if updated_imports or updated_training:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("-> Saved modified TF-IDF notebook successfully.")
    else:
        print("-> TF-IDF notebook already updated or match not found.")

def update_transformer_notebook(path):
    print(f"Modifying Transformer notebook: {path}...")
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return
        
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    updated_imports = False
    updated_args = False
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # 1. Update imports cell to include mlflow
            if "from transformers import AutoTokenizer, AutoModelForSequenceClassification" in source and "import mlflow" not in source:
                cell['source'] = [
                    "import torch\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import ast\n",
                    "from sklearn.preprocessing import MultiLabelBinarizer\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "from datasets import Dataset, DatasetDict\n",
                    "from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer\n",
                    "from sklearn.metrics import f1_score, accuracy_score\n",
                    "import mlflow\n",
                    "\n",
                    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
                    "print(f\"Using device: {device}\")\n",
                    "\n",
                    "# Set the MLflow experiment name\n",
                    "mlflow.set_experiment(\"transformer_triage\")\n"
                ]
                updated_imports = True
                print("-> Updated imports cell.")
            
            # 2. Update training arguments to enable mlflow reporting
            elif "training_args = TrainingArguments(" in source and "report_to=" not in source:
                cell['source'] = [
                    "# 1. Load the pre-trained model\n",
                    "model = AutoModelForSequenceClassification.from_pretrained(\n",
                    "    model_name, \n",
                    "    num_labels=num_labels, \n",
                    "    problem_type=\"multi_label_classification\"\n",
                    ")\n",
                    "\n",
                    "# 2. Define training rules\n",
                    "training_args = TrainingArguments(\n",
                    "    output_dir=\"./ticket_triage_model\",\n",
                    "    eval_strategy=\"epoch\",    # Check performance at the end of every epoch\n",
                    "    save_strategy=\"epoch\",\n",
                    "    learning_rate=2e-5,             # Standard learning rate for fine-tuning\n",
                    "    per_device_train_batch_size=16, # Fits safely in 8GB VRAM\n",
                    "    per_device_eval_batch_size=16,\n",
                    "    num_train_epochs=3,             # 3 passes over the data is usually plenty\n",
                    "    weight_decay=0.01,\n",
                    "    fp16=True,                      # RTX 4060 Magic! Makes training super fast\n",
                    "    load_best_model_at_end=True,\n",
                    "    report_to=\"mlflow\",             # Log results to MLflow\n",
                    ")\n",
                    "\n",
                    "# 3. Initialize the Trainer\n",
                    "trainer = Trainer(\n",
                    "    model=model,\n",
                    "    args=training_args,\n",
                    "    train_dataset=tokenized_datasets[\"train\"],\n",
                    "    eval_dataset=tokenized_datasets[\"test\"],\n",
                    "    processing_class=tokenizer,\n",
                    "    compute_metrics=compute_metrics\n",
                    ")\n",
                    "\n",
                    "# 4. START TRAINING!\n",
                    "print(\"Starting training...\")\n",
                    "trainer.train()\n"
                ]
                updated_args = True
                print("-> Updated training arguments cell.")

    if updated_imports or updated_args:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("-> Saved modified Transformer notebook successfully.")
    else:
        print("-> Transformer notebook already updated or match not found.")

if __name__ == "__main__":
    update_tfidf_notebook("notebooks/tfidf-baseline.ipynb")
    update_transformer_notebook("notebooks/transformer_models.ipynb")
