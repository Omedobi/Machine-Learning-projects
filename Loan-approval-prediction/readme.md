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
zenml stack update -d mlflow ~use to update deployer
zenml stack delete default ~use to delete an active stack~
```

```bash
zenml integration install mlflow
print(Client().active_stack.experiment_tracker.get_tracking_url()) #this enable you to get the tracking url
```


```bash
- getting the url of the experiment_tracker to run locally.
mlflow ui --backend-store-uri "file:C:\Users\admin\AppData\Roaming\zenml\local_stores\0c1099e2-92b6-41fe-9381-f674eeac16ea\mlruns"
```
"-> : means return"

zenml down --blocking

pip cache purge