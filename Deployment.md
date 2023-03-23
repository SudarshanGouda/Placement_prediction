# Deployment on Azure Cloud

## Set up your initial environment
<ol>
<li>Have an Azure account with an active subscription. <a href="https://azure.microsoft.com/free/?ref=microsoft.com&amp;utm_source=microsoft.com&amp;utm_medium=docs&amp;utm_campaign=visualstudio" data-linktype="external">Create an account for free</a>.</li>
<li>Install <a href="https://www.python.org/downloads/" target="_blank" data-linktype="external">Python 3.6 or higher</a>.</li>
<li>Install the <a href="/en-us/cli/azure/install-azure-cli" target="_blank" data-linktype="absolute-path">Azure CLI</a>, with which you run commands in any shell to provision and configure Azure resources.</li>
</ol>

## Commands for the CLI 

<br>

##### Check the Python Version 

```
 py -3 --version

```

##### Check the Azure CLI Version 

```
az --version

```

##### Then sign in to Azure through the CLI:

```
az login

```
 A tab will be opened in the browser,login into you're registered azure account,


Post successfull login,go to the folder where the code files are present,using the `cd` command as shown ↓
 

```
cd <project folder>

```

Once the control is inside the project folder,you need to install all the requirements which are mentioned in the `requirements.txt` file,but before that we need to create a virtual environment!,it is very important,so for creating a `virtual environment` follow the commands 

```
py -3 -m venv .venv
```

The above ↑ command will create a virtual environment,then type in 

```
.venv\scripts\activate
```

This command is to activate the virtual environment,after this type in 

```
pip install -r requirements.txt
```

By this command you will install all the packages which are mentioned in the `requirements.txt` file,into the virtual environment,once this is done,the task is almost 80% completed,after this inorder to check if the `Flask` App is running successfully on local host run this command

```
flask run
```
If you face any error while running this command,check out whether you're in the same directory where your `app.py` is residing or not,Once this command is successfully executed,the last step is to deploy the flask app on the `Azure` Platform,so for that follow the following command ↓

```
az webapp up --sku B1 --name <app-name>
```

This will take sometime,after completion of this command,you'll find something like this ↓ in your cmd (for example!)

```
{
  "URL": "https://campusplacementpredictor.azurewebsites.net/",
  
}
```
Copy that url from `URL` and that is your deployment of the web app on the Azure cloud.


