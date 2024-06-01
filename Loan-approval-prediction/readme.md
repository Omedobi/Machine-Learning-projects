# Machine Learning Ops Projects
```bash
git clone https://github.com/Omedobi/Data-Science-Projects.git
cd Data-Science-Projects/ML-Ops/loan Prediction
pip install -r Requirements.txt
```

```bash
pip install "zenml[server]"
zenml up --blocking
zenml integration install mlflow -y
zenml stack list
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
zenml stack describe
zenml model-deployer register mlflow --flavor=mlflow

```

```bash
zenml integration install mlflow
print(Client().active_stack.experiment_tracker.get_tracking_url()) #this enable you to get the tracking url
```


```bash
- getting the url of the experiment_tracker to run locally.
mlflow ui --backend-store-uri "file:C:\Users\admin\AppData\Roaming\zenml\local_stores\d6b37deb-d883-4111-8733-a9d0285942b5\mlruns"
```
"-> : means return"