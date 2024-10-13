# AlzPrevent

Este é um projeto Django para o sistema **AlzPrevent**, que tem como objetivo prevenir e analisar fatores de risco e associações causais relacionadas à Doença de Alzheimer.

## Pré-requisitos

Antes de rodar o projeto, você precisará das seguintes ferramentas instaladas em sua máquina:

- [Python 3.x](https://www.python.org/downloads/)
- [Pip](https://pip.pypa.io/en/stable/installation/)
- [Virtualenv](https://virtualenv.pypa.io/en/stable/installation/)
- [Django](https://www.djangoproject.com/)

## Como rodar o projeto

### 1. Clonar o repositório

```bash
git clone https://github.com/luiiizz/AlzPrevent.git
cd AlzPrevent/projectAlzPrevent
```

### 2. Realizar as migrações do banco de dados

```bash
python manage.py migrate
```

### 3. Instalar a biblioteca XGBoost

```bash
pip install xgboost==2.1.1
```

### 4. Rodar o servidor de desenvolvimento

```bash
python manage.py runserver
```

### 5. Acesse o projeto no navegador

```
http://127.0.0.1:8000/

```
