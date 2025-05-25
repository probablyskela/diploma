```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python book_storage_project/manage.py migrate
python book_storage_project/manage.py runserver
```

Add ukrainian stop words to nltk_data directory.
