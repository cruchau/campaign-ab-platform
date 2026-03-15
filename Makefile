setup:
	python3 -m venv ~/.campaign_lift
	source ~/.campaign_lift/bin/activate 

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt